#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>

void input(char *input_filename);
void output(char *output_filename);

__global__ void flash_attention_kernel(float* q, float* k, float* v, float* o, int B, int N, int d, int tr, int tc, int br, int bc);

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    float *d_Q, *d_K, *d_V, *d_O;
    size_t size = B * N * d * sizeof(float);
    hipMalloc(&d_Q, size);
    hipMalloc(&d_K, size);
    hipMalloc(&d_V, size);
    hipMalloc(&d_O, size);

    hipMemcpy(d_Q, Q, size, hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, size, hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, size, hipMemcpyHostToDevice);
    hipMemset(d_O, 0, size);

    int br = 128;
    int bc = 32;
    
    int tr = (N + br - 1) / br;
    int tc = (N + bc - 1) / bc;

    dim3 block(br / 4 * 32); 
    dim3 grid(B * tr);
    size_t s_mem_size = (4 * bc * d) * sizeof(float);

    double start, end;
    start = getTimeStamp();

    flash_attention_kernel<<<grid, block, s_mem_size>>>(d_Q, d_K, d_V, d_O, B, N, d, tr, tc, br, bc);
    hipDeviceSynchronize();

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    hipMemcpy(O, d_O, size, hipMemcpyDeviceToHost);

    output(argv[2]);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

__global__ void __launch_bounds__(1024) flash_attention_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v, float* __restrict__ o, int B, int N, int d, int tr, int tc, int br, int bc) {
    extern __shared__ __align__(16) float s_mem[];
    float* s_K_base = s_mem;
    float* s_V_base = s_mem + 2 * bc * d;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int warpId = tx / 32;
    int laneId = tx % 32;

    int tile_idx = bx % tr;
    int batch_id = bx / tr;
    
    int group_id = laneId / 16; 
    int lane_in_group = laneId % 16;
    
    int row_base = tile_idx * br + warpId * 4;
    int my_row_0 = row_base + group_id;      
    int my_row_1 = row_base + group_id + 2;  
    
    float4 q_vec[2];
    float4 o_vec[2];
    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0.0f, 0.0f};
    
    o_vec[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    o_vec[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    float scale = rsqrtf((float)d);
    int d_div_4 = d / 4;
    
    float4* Q_f4 = (float4*)q;
    int q_offset_base = batch_id * N * d_div_4;
    
    if (my_row_0 < N && lane_in_group < d_div_4)
        q_vec[0] = Q_f4[q_offset_base + my_row_0 * d_div_4 + lane_in_group];
    else
        q_vec[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
    if (my_row_1 < N && lane_in_group < d_div_4)
        q_vec[1] = Q_f4[q_offset_base + my_row_1 * d_div_4 + lane_in_group];
    else
        q_vec[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Prologue: Load tile 0
    {
        int j = 0;
        int kv_base = batch_id * N * d + j * bc * d;
        int total_elements = bc * d;
        
        float4* s_K_vec = (float4*)(s_K_base + (j % 2) * bc * d);
        float4* s_V_vec = (float4*)(s_V_base + (j % 2) * bc * d);
        float4* k_vec_g = (float4*)(k + kv_base);
        float4* v_vec_g = (float4*)(v + kv_base);
        
        int num_vec = total_elements / 4;
        for (int i = tx; i < num_vec; i += blockDim.x) {
            int row = (i * 4) / d;
            if (j * bc + row < N) {
                s_K_vec[i] = k_vec_g[i];
                s_V_vec[i] = v_vec_g[i];
            } else {
                s_K_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                s_V_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
    
    __syncthreads();

    for (int j = 0; j < tc; j++) {
        // Prefetch tile j+1
        if (j < tc - 1) {
            int next_j = j + 1;
            int kv_base = batch_id * N * d + next_j * bc * d;
            int total_elements = bc * d;
            
            float4* s_K_vec = (float4*)(s_K_base + (next_j % 2) * bc * d);
            float4* s_V_vec = (float4*)(s_V_base + (next_j % 2) * bc * d);
            float4* k_vec_g = (float4*)(k + kv_base);
            float4* v_vec_g = (float4*)(v + kv_base);
            
            int num_vec = total_elements / 4;
            for (int i = tx; i < num_vec; i += blockDim.x) {
                int row = (i * 4) / d;
                if (next_j * bc + row < N) {
                    s_K_vec[i] = k_vec_g[i];
                    s_V_vec[i] = v_vec_g[i];
                } else {
                    s_K_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    s_V_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }

        float4* s_K_f4 = (float4*)(s_K_base + (j % 2) * bc * d);
        float4* s_V_f4 = (float4*)(s_V_base + (j % 2) * bc * d);

        for (int t = 0; t < bc; t++) {
            if (j * bc + t >= N) break;

            float4 k_val;
            if (lane_in_group < d_div_4)
                k_val = s_K_f4[t * d_div_4 + lane_in_group];
            else
                k_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            float score_0 = q_vec[0].x * k_val.x + q_vec[0].y * k_val.y + q_vec[0].z * k_val.z + q_vec[0].w * k_val.w;
            float score_1 = q_vec[1].x * k_val.x + q_vec[1].y * k_val.y + q_vec[1].z * k_val.z + q_vec[1].w * k_val.w;
            
            for (int offset = 8; offset > 0; offset /= 2) {
                score_0 += __shfl_down(score_0, offset, 32);
                score_1 += __shfl_down(score_1, offset, 32);
            }
            
            int leader = group_id * 16;
            score_0 = __shfl(score_0, leader, 32);
            score_1 = __shfl(score_1, leader, 32);
            
            score_0 *= scale;
            score_1 *= scale;
            
            float m_prev = m[0];
            float m_new = fmaxf(m_prev, score_0);
            float exp_s = __expf(score_0 - m_new);
            float exp_m = __expf(m_prev - m_new);
            l[0] = l[0] * exp_m + exp_s;
            o_vec[0].x = o_vec[0].x * exp_m;
            o_vec[0].y = o_vec[0].y * exp_m;
            o_vec[0].z = o_vec[0].z * exp_m;
            o_vec[0].w = o_vec[0].w * exp_m;
            m[0] = m_new;
            
            m_prev = m[1];
            m_new = fmaxf(m_prev, score_1);
            exp_s = __expf(score_1 - m_new);
            exp_m = __expf(m_prev - m_new);
            l[1] = l[1] * exp_m + exp_s;
            o_vec[1].x = o_vec[1].x * exp_m;
            o_vec[1].y = o_vec[1].y * exp_m;
            o_vec[1].z = o_vec[1].z * exp_m;
            o_vec[1].w = o_vec[1].w * exp_m;
            m[1] = m_new;
            
            float4 v_val;
            if (lane_in_group < d_div_4)
                v_val = s_V_f4[t * d_div_4 + lane_in_group];
            else
                v_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                
            float p0 = __expf(score_0 - m[0]);
            o_vec[0].x += p0 * v_val.x;
            o_vec[0].y += p0 * v_val.y;
            o_vec[0].z += p0 * v_val.z;
            o_vec[0].w += p0 * v_val.w;
            
            float p1 = __expf(score_1 - m[1]);
            o_vec[1].x += p1 * v_val.x;
            o_vec[1].y += p1 * v_val.y;
            o_vec[1].z += p1 * v_val.z;
            o_vec[1].w += p1 * v_val.w;
        }
        __syncthreads();
    }

    float4* O_f4 = (float4*)o;
    int o_offset_base = batch_id * N * d_div_4;
    
    if (my_row_0 < N && lane_in_group < d_div_4) {
        float inv_l = 1.0f / l[0];
        float4 res;
        res.x = o_vec[0].x * inv_l;
        res.y = o_vec[0].y * inv_l;
        res.z = o_vec[0].z * inv_l;
        res.w = o_vec[0].w * inv_l;
        O_f4[o_offset_base + my_row_0 * d_div_4 + lane_in_group] = res;
    }
    
    if (my_row_1 < N && lane_in_group < d_div_4) {
        float inv_l = 1.0f / l[1];
        float4 res;
        res.x = o_vec[1].x * inv_l;
        res.y = o_vec[1].y * inv_l;
        res.z = o_vec[1].z * inv_l;
        res.w = o_vec[1].w * inv_l;
        O_f4[o_offset_base + my_row_1 * d_div_4 + lane_in_group] = res;
    }
}
