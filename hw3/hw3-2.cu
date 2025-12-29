#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

#define INF 1073741823  // (1<<30)-1
#define USHORT_MAX 65535

// ======================================================
// Pre/Post Processing Kernels
// ======================================================

__global__ void init_dist_kernel(ushort* Dist, int n, int nPad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nPad && idy < nPad) {
        if (idx < n && idy < n) {
             Dist[idy * nPad + idx] = (idx == idy) ? 0 : USHORT_MAX;
        } else {
             Dist[idy * nPad + idx] = USHORT_MAX;
        }
    }
}

__global__ void update_edges_kernel(ushort* Dist, int* edges, int m, int nPad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int u = edges[3 * idx + 0];
        int v = edges[3 * idx + 1];
        int w = edges[3 * idx + 2];
        Dist[u * nPad + v] = (ushort)w;
    }
}

__global__ void dist_to_out_kernel(ushort* Dist, int* Out, int n, int nPad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        Out[idy * n + idx] = (int)Dist[idy * nPad + idx];
    }
}

// 已移除 CUDA_CHECK，直接呼叫 CUDA runtime API（未檢查錯誤）


// ======================================================
// Phase Kernels
// ======================================================

// Phase 1: pivot block (r, r)
// 改成 64x64 tile 使用 32x32 threads；每 thread 處理 2x2 微分塊 (4 元素)
__global__ void phase1(ushort* Dist, int nPad, int B, int r)
{
    int tyi = threadIdx.y; // 0..31
    int txj = threadIdx.x; // 0..31
    int off_i = (tyi << 1); // local row start (0..62 step 2)
    int off_j = (txj << 1); // local col start (0..62 step 2)

    int row0 = r * B + off_i;
    int row1 = row0 + 1;
    int col0 = r * B + off_j;
    int col1 = col0 + 1;

    const int STRIDE = B + 1; // shared padding stride to avoid 32-bank conflicts
    __shared__ int pivot[64 * (64 + 1)];

    // 載入四元素 (所有 threads 合作載完整 64x64)
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
// gridDim = (numBlocks-1, 2)
//   y=0 → row blocks (r, coord)
//   y=1 → col blocks (coord, r)
__global__ void phase2(
    ushort* Dist,
    int  nPad,
    int  B,
    int  r,
    int  numBlocks
){
    int tyi = threadIdx.y; // 0..31
    int txj = threadIdx.x; // 0..31
    int off_i = (tyi << 1);
    int off_j = (txj << 1);

    const int STRIDE2 = B + 1;
    __shared__ int tileA[64 * (64 + 1)];
    __shared__ int tileB[64 * (64 + 1)];

    int which = blockIdx.y; // 0=row, 1=col
    int t = blockIdx.x;
    int coord = (t < r) ? t : t + 1;
    int bi = (which == 0 ? r : coord);
    int bj = (which == 0 ? coord : r);

    int base_i = bi * B + off_i;
    int base_j = bj * B + off_j;

    // load 2x2 micro-tile from Dist into shared tileA
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int gi = base_i + di;
            int gj = base_j + dj;
            if (gi < nPad && gj < nPad)
                tileA[(off_i + di) * STRIDE2 + (off_j + dj)] = Dist[gi * nPad + gj];
        }
    }
    // load pivot block (r,r) into tileB
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

// Phase 3: other blocks (neither in row r nor col r)
// gridDim = (numBlocks, numBlocks)
__global__ void phase3(
    ushort* __restrict__ Dist,
    int  nPad,
    int  B,
    int  r,
    int  numBlocks
){
    // thread mapping: each thread updates a 2x2 micro-tile
    int ty = threadIdx.y; // 0..31
    int tx = threadIdx.x; // 0..31
    int off_i = (ty << 1);
    int off_j = (tx << 1);

    int block_i = blockIdx.y;
    int block_j = blockIdx.x;

    if (block_i == r || block_j == r) return;

    // ===== SLICE SIZE = 64 (no slicing needed) =====
    const int SLICE_K = 64;

    // ===== Shared memory: FULL pivot row + pivot column =====
    __shared__ uint pivotRowSlice[SLICE_K][SLICE_K]; // 64 x 64
    __shared__ uint pivotColSlice[SLICE_K][SLICE_K]; // 64 x 64

    int base_i0 = block_i * B;
    int base_j0 = block_j * B;

    int row0 = base_i0 + off_i;
    int row1 = row0 + 1;
    int col0 = base_j0 + off_j;
    int col1 = col0 + 1;

    // === preload Dist[row][col] into registers ===
    uint best00 = Dist[row0 * nPad + col0];
    uint best01 = Dist[row0 * nPad + col1];
    uint best10 = Dist[row1 * nPad + col0];
    uint best11 = Dist[row1 * nPad + col1];

    // ============================================================
    // Load FULL pivot row slice (64x64) into shared memory
    // ============================================================
    for (int kb = 0; kb < SLICE_K; kb += 32) {
        int prow = r * B + kb + ty;
        for (int cb = 0; cb < SLICE_K; cb += 32) {
            int gj = base_j0 + cb + tx;
            uint v = (prow < nPad && gj < nPad) ? Dist[prow * nPad + gj] : USHORT_MAX;
            pivotRowSlice[kb + ty][cb + tx] = v;
        }
    }

    // ============================================================
    // Load FULL pivot column slice (64x64) into shared memory
    // ============================================================
    for (int rb = 0; rb < SLICE_K; rb += 32) {
        int grow = base_i0 + rb + ty;
        for (int kb = 0; kb < SLICE_K; kb += 32) {
            int pcol = r * B + kb + tx;
            uint v = (grow < nPad && pcol < nPad) ? Dist[grow * nPad + pcol] : USHORT_MAX;
            pivotColSlice[rb + ty][kb + tx] = v;
        }
    }

    __syncthreads(); // shared pivot fully ready

    // ============================================================
    // FULL kLocal loop: 0..63 (unrolled)
    // ============================================================
    #pragma unroll 64
    for (int kLocal = 0; kLocal < SLICE_K; ++kLocal) {

        uint w_row0_k = pivotColSlice[off_i][kLocal];
        uint w_row1_k = pivotColSlice[off_i + 1][kLocal];
        uint w_k_col0 = pivotRowSlice[kLocal][off_j];
        uint w_k_col1 = pivotRowSlice[kLocal][off_j + 1];

        uint via00 = w_row0_k + w_k_col0;
        uint via01 = w_row0_k + w_k_col1;
        uint via10 = w_row1_k + w_k_col0;
        uint via11 = w_row1_k + w_k_col1;

        best00 = min(best00, via00);
        best01 = min(best01, via01);
        best10 = min(best10, via10);
        best11 = min(best11, via11);
    }

    // ============================================================
    // Write back results
    // ============================================================
    Dist[row0 * nPad + col0] = best00;
    Dist[row0 * nPad + col1] = best01;
    Dist[row1 * nPad + col0] = best10;
    Dist[row1 * nPad + col1] = best11;
}

// ======================================================
// Input / Output（1D 展開）
// ======================================================
void input(char* infile, int B, ushort** Dist_ptr, int* n_ptr, int* nPad_ptr){
    FILE* file = fopen(infile, "rb");
    if (!file) {
        perror("fopen input");
        exit(1);
    }

    int n, m;
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    *n_ptr = n;

    int nPad = ((n + B - 1) / B) * B;
    *nPad_ptr = nPad;

    ushort* Dist_d = NULL;
    cudaMalloc((void**)&Dist_d, nPad * nPad * sizeof(ushort));

    // Initialize Dist on Device
    dim3 block(32, 32);
    dim3 grid((nPad + 31) / 32, (nPad + 31) / 32);
    init_dist_kernel<<<grid, block>>>(Dist_d, n, nPad);

    // Read edges
    int* edges_h = (int*)malloc(m * 3 * sizeof(int));
    fread(edges_h, sizeof(int), 3 * m, file);
    fclose(file);

    // Copy edges to Device
    int* edges_d;
    cudaMalloc((void**)&edges_d, m * 3 * sizeof(int));
    cudaMemcpy(edges_d, edges_h, m * 3 * sizeof(int), cudaMemcpyHostToDevice);

    // Update Dist with edges
    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;
    update_edges_kernel<<<numBlocks, blockSize>>>(Dist_d, edges_d, m, nPad);

    cudaFree(edges_d);
    free(edges_h);

    *Dist_ptr = Dist_d;
}

void output(char* outfile, ushort* Dist_d, int n, int nPad) {
    int fd = open(outfile, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) { perror("open output"); exit(1); }

    size_t size = (size_t)n * n * sizeof(int);
    if (ftruncate(fd, size) != 0) { perror("ftruncate"); exit(1); }

    int* map_ptr = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) { perror("mmap"); exit(1); }

    // Register mmap memory as pinned memory for CUDA
    cudaHostRegister(map_ptr, size, cudaHostRegisterDefault);

    int* Out_d;
    cudaMalloc((void**)&Out_d, size);

    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (n + 31) / 32);
    dist_to_out_kernel<<<grid, block>>>(Dist_d, Out_d, n, nPad);

    cudaMemcpyAsync(map_ptr, Out_d, size, cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    cudaFree(Out_d);
    cudaHostUnregister(map_ptr);
    munmap(map_ptr, size);
    close(fd);
}


// ======================================================
// Host: Blocked FW kernel launcher
// ======================================================
void block_FW_CUDA(ushort* Dist_d, int n, int nPad, int B){
    int rounds    = nPad / B; // 一邊有多少個 blocks

    // 使用 32x32 threads 映射到 64x64 tile (2x2 micro-tiles)
    dim3 blockDim(32, 32);

    for (int r = 0; r < rounds; ++r) {
        // ===== Phase 1: pivot block (r,r) =====
        dim3 gridPhase1(1, 1);
        phase1<<<gridPhase1, blockDim>>>(Dist_d, nPad, B, r);
            
        // ===== Phase 2: pivot row & column =====
        dim3 gridPhase2(rounds - 1, 2);   // x = all blocks except r, y = 0(row),1(col)
        phase2<<<gridPhase2, blockDim>>>(Dist_d, nPad, B, r, rounds);

        // ===== Phase 3: remaining blocks =====
        dim3 gridPhase3(rounds, rounds);
        phase3<<<gridPhase3, blockDim>>>(Dist_d, nPad, B, r, rounds);
    }
}


// ======================================================
// Main
// ======================================================
int main(int argc, char** argv){
    if (argc < 3) {
        printf("Usage: %s input.bin output.bin\n", argv[0]);
        return 0;
    }

    ushort *Dist_d;
    int n, nPad;
    int B = 64;  // 64x64 tile；32x32 threads，每 thread 處理 2x2 微分塊

    nvtxRangePush("input");
    input(argv[1], B, &Dist_d, &n, &nPad);
    nvtxRangePop();
    block_FW_CUDA(Dist_d, n, nPad, B);
    nvtxRangePush("output");
    output(argv[2], Dist_d, n, nPad);
    nvtxRangePop(); 
    cudaFree(Dist_d);
    return 0;
}