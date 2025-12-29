#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define INF 1073741823
#define USHORT_MAX 65535

// ======================================================
// Phase Kernels
// ======================================================

// Phase 1: pivot block (r, r)
__global__ void phase1(unsigned short* Dist, int nPad, int B, int r)
{
    int tyi = threadIdx.y;
    int txj = threadIdx.x;
    int off_i = (tyi << 1);
    int off_j = (txj << 1);

    int row0 = r * B + off_i;
    int row1 = row0 + 1;
    int col0 = r * B + off_j;
    int col1 = col0 + 1;

    const int STRIDE = B + 1;
    __shared__ int pivot[64 * (64 + 1)];

    if (row0 < nPad && col0 < nPad) pivot[off_i * STRIDE + off_j] = Dist[row0 * nPad + col0];
    if (row0 < nPad && col1 < nPad) pivot[off_i * STRIDE + off_j + 1] = Dist[row0 * nPad + col1];
    if (row1 < nPad && col0 < nPad) pivot[(off_i + 1) * STRIDE + off_j] = Dist[row1 * nPad + col0];
    if (row1 < nPad && col1 < nPad) pivot[(off_i + 1) * STRIDE + off_j + 1] = Dist[row1 * nPad + col1];

    __syncthreads();

    for (int k = 0; k < B; ++k) {
        int r0k = pivot[off_i * STRIDE + k];
        int r1k = pivot[(off_i + 1) * STRIDE + k];
        int kc0 = pivot[k * STRIDE + off_j];
        int kc1 = pivot[k * STRIDE + off_j + 1];

        int via00 = (r0k == INF || kc0 == INF) ? INF : r0k + kc0;
        int via01 = (r0k == INF || kc1 == INF) ? INF : r0k + kc1;
        int via10 = (r1k == INF || kc0 == INF) ? INF : r1k + kc0;
        int via11 = (r1k == INF || kc1 == INF) ? INF : r1k + kc1;

        int idx00 = off_i * STRIDE + off_j;
        int idx01 = off_i * STRIDE + off_j + 1;
        int idx10 = (off_i + 1) * STRIDE + off_j;
        int idx11 = (off_i + 1) * STRIDE + off_j + 1;

        if (via00 < pivot[idx00]) pivot[idx00] = via00;
        if (via01 < pivot[idx01]) pivot[idx01] = via01;
        if (via10 < pivot[idx10]) pivot[idx10] = via10;
        if (via11 < pivot[idx11]) pivot[idx11] = via11;

        __syncthreads();
    }

    if (row0 < nPad && col0 < nPad) Dist[row0 * nPad + col0] = pivot[off_i * STRIDE + off_j];
    if (row0 < nPad && col1 < nPad) Dist[row0 * nPad + col1] = pivot[off_i * STRIDE + off_j + 1];
    if (row1 < nPad && col0 < nPad) Dist[row1 * nPad + col0] = pivot[(off_i + 1) * STRIDE + off_j];
    if (row1 < nPad && col1 < nPad) Dist[row1 * nPad + col1] = pivot[(off_i + 1) * STRIDE + off_j + 1];
}

// Phase 2: pivot row & pivot column blocks
__global__ void phase2(
    unsigned short* Dist,
    int nPad,
    int B,
    int r,
    int numBlocks
){
    int tyi = threadIdx.y;
    int txj = threadIdx.x;
    int off_i = (tyi << 1);
    int off_j = (txj << 1);

    const int STRIDE2 = B + 1;
    __shared__ int tileA[64 * (64 + 1)];
    __shared__ int tileB[64 * (64 + 1)];

    int which = blockIdx.y;
    int t = blockIdx.x;
    int coord = (t < r) ? t : t + 1;
    int bi = (which == 0 ? r : coord);
    int bj = (which == 0 ? coord : r);

    int base_i = bi * B + off_i;
    int base_j = bj * B + off_j;

    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int gi = base_i + di;
            int gj = base_j + dj;
            if (gi < nPad && gj < nPad)
                tileA[(off_i + di) * STRIDE2 + (off_j + dj)] = Dist[gi * nPad + gj];
        }
    }

    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int gi = r * B + off_i + di;
            int gj = r * B + off_j + dj;
            if (gi < nPad && gj < nPad)
                tileB[(off_i + di) * STRIDE2 + (off_j + dj)] = Dist[gi * nPad + gj];
        }
    }

    __syncthreads();

    for (int k = 0; k < B; ++k) {
        for (int di = 0; di < 2; ++di) {
            int rowLocal = off_i + di;
            int pivot_i_k = (which == 0) ? tileB[rowLocal * STRIDE2 + k] : tileA[rowLocal * STRIDE2 + k];
            for (int dj = 0; dj < 2; ++dj) {
                int colLocal = off_j + dj;
                int other_k_j = (which == 0) ? tileA[k * STRIDE2 + colLocal] : tileB[k * STRIDE2 + colLocal];
                int via = (pivot_i_k == INF || other_k_j == INF) ? INF : (pivot_i_k + other_k_j);
                int idx = rowLocal * STRIDE2 + colLocal;
                if (via < tileA[idx]) tileA[idx] = via;
            }
        }
        __syncthreads();
    }

    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int gi = base_i + di;
            int gj = base_j + dj;
            if (gi < nPad && gj < nPad)
                Dist[gi * nPad + gj] = tileA[(off_i + di) * STRIDE2 + (off_j + dj)];
        }
    }
}

// Phase 3: other blocks
__global__ void phase3(
    unsigned short* __restrict__ Dist,
    int nPad,
    int B,
    int r,
    int numBlocks
){
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int off_i = (ty << 1);
    int off_j = (tx << 1);

    int block_i = blockIdx.y;
    int block_j = blockIdx.x;

    if (block_i == r || block_j == r) return;

    const int SLICE_K = 64;

    __shared__ unsigned int pivotRowSlice[SLICE_K][SLICE_K];
    __shared__ unsigned int pivotColSlice[SLICE_K][SLICE_K];

    int base_i0 = block_i * B;
    int base_j0 = block_j * B;

    int row0 = base_i0 + off_i;
    int row1 = row0 + 1;
    int col0 = base_j0 + off_j;
    int col1 = col0 + 1;

    unsigned int best00 = Dist[row0 * nPad + col0];
    unsigned int best01 = Dist[row0 * nPad + col1];
    unsigned int best10 = Dist[row1 * nPad + col0];
    unsigned int best11 = Dist[row1 * nPad + col1];

    for (int kb = 0; kb < SLICE_K; kb += 32) {
        int prow = r * B + kb + ty;
        for (int cb = 0; cb < SLICE_K; cb += 32) {
            int gj = base_j0 + cb + tx;
            unsigned int v = (prow < nPad && gj < nPad) ? Dist[prow * nPad + gj] : USHORT_MAX;
            pivotRowSlice[kb + ty][cb + tx] = v;
        }
    }

    for (int rb = 0; rb < SLICE_K; rb += 32) {
        int grow = base_i0 + rb + ty;
        for (int kb = 0; kb < SLICE_K; kb += 32) {
            int pcol = r * B + kb + tx;
            unsigned int v = (grow < nPad && pcol < nPad) ? Dist[grow * nPad + pcol] : USHORT_MAX;
            pivotColSlice[rb + ty][kb + tx] = v;
        }
    }

    __syncthreads();

    #pragma unroll 64
    for (int kLocal = 0; kLocal < SLICE_K; ++kLocal) {
        unsigned int w_row0_k = pivotColSlice[off_i][kLocal];
        unsigned int w_row1_k = pivotColSlice[off_i + 1][kLocal];
        unsigned int w_k_col0 = pivotRowSlice[kLocal][off_j];
        unsigned int w_k_col1 = pivotRowSlice[kLocal][off_j + 1];

        unsigned int via00 = w_row0_k + w_k_col0;
        unsigned int via01 = w_row0_k + w_k_col1;
        unsigned int via10 = w_row1_k + w_k_col0;
        unsigned int via11 = w_row1_k + w_k_col1;

        best00 = min(best00, via00);
        best01 = min(best01, via01);
        best10 = min(best10, via10);
        best11 = min(best11, via11);
    }

    Dist[row0 * nPad + col0] = best00;
    Dist[row0 * nPad + col1] = best01;
    Dist[row1 * nPad + col0] = best10;
    Dist[row1 * nPad + col1] = best11;
}

// ======================================================
// Input / Output
// ======================================================
void input(char* infile, int B, unsigned short** Dist_ptr, int* n_ptr, int* nPad_ptr){
    FILE* file = fopen(infile, "rb");
    if (!file) exit(1);

    int n, m;
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    *n_ptr = n;
    int nPad = ((n + B - 1) / B) * B;
    *nPad_ptr = nPad;

    unsigned short* Dist = NULL;
    hipHostMalloc((void**)&Dist, nPad * nPad * sizeof(unsigned short));

    for (int i = 0; i < nPad; ++i) {
        for (int j = 0; j < nPad; ++j) {
            if (i < n && j < n) {
                Dist[i*nPad + j] = (i == j ? 0 : USHORT_MAX);
            } else {
                Dist[i*nPad + j] = USHORT_MAX;
            }
        }
    }

    int edge[3];
    for (int i = 0; i < m; ++i) {
        fread(edge, sizeof(int), 3, file);
        int u = edge[0], v = edge[1], w = edge[2];
        Dist[u*nPad + v] = w;
    }

    fclose(file);
    *Dist_ptr = Dist;
}

void output(char* outfile, unsigned short* Dist, int n, int nPad){
    FILE* out = fopen(outfile, "wb");
    if (!out) exit(1);

    int* rowBuf = (int*) malloc(n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            rowBuf[j] = (int) Dist[i*nPad + j];
        }
        fwrite(rowBuf, sizeof(int), n, out);
    }

    free(rowBuf);
    fclose(out);
}


// ======================================================
// Host Launcher
// ======================================================
void block_FW_HIP(unsigned short* Dist, int n, int nPad, int B){
    int rounds = nPad / B;
    int numBlocks = rounds;

    unsigned short *Dist_d = NULL;
    hipMalloc(&Dist_d, nPad * nPad * sizeof(unsigned short));
    hipMemcpy(Dist_d, Dist, nPad*nPad*sizeof(unsigned short), hipMemcpyHostToDevice);

    dim3 blockDim(32, 32);

    for (int r = 0; r < rounds; ++r) {
        dim3 gridPhase1(1, 1);
        phase1<<<gridPhase1, blockDim>>>(Dist_d, nPad, B, r);

        dim3 gridPhase2(rounds - 1, 2);
        dim3 threadsPhase2(32, 32);
        phase2<<<gridPhase2, threadsPhase2>>>(Dist_d, nPad, B, r, rounds);

        dim3 gridPhase3(rounds, rounds);
        phase3<<<gridPhase3, blockDim>>>(Dist_d, nPad, B, r, rounds);
    }

    hipMemcpy(Dist, Dist_d, nPad*nPad*sizeof(unsigned short), hipMemcpyDeviceToHost);
    hipFree(Dist_d);
}


// ======================================================
// Main
// ======================================================
int main(int argc, char** argv){
    if (argc < 3) {
        printf("Usage: %s input.bin output.bin\n", argv[0]);
        return 0;
    }

    unsigned short *Dist;
    int n, nPad;
    int B = 64;

    input(argv[1], B, &Dist, &n, &nPad);
    block_FW_HIP(Dist, n, nPad, B);
    output(argv[2], Dist, n, nPad);

    hipHostFree(Dist);
    return 0;
}