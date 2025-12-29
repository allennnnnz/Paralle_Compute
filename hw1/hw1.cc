#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <algorithm>
#define BOOST_SPREADSORT_MAX_SPLITS 12
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cstring>
#include <numa.h>
#include <sched.h>

// 合併保留最小 k 個（輸出寫回 buf）
inline void merge_keep_smallest_inplace(float *a, size_t n1,
                                        const float *b, size_t n2,
                                        float *buf, size_t k) {
    size_t i=0, j=0, t=0;
    while (t < k && i < n1 && j < n2)
        buf[t++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (t < k && i < n1) buf[t++] = a[i++];
    while (t < k && j < n2) buf[t++] = b[j++];
}

// 合併保留最大 k 個（輸出寫回 buf）
inline void merge_keep_largest_inplace(float *a, size_t n1,
                                       const float *b, size_t n2,
                                       float *buf, size_t k) {
    size_t i=n1, j=n2;
    size_t t=k;
    while (t>0 && i>0 && j>0)
        buf[--t] = (a[i-1] >= b[j-1]) ? a[--i] : b[--j];
    while (t>0 && i>0) buf[--t] = a[--i];
    while (t>0 && j>0) buf[--t] = b[--j];
}

int main(int argc, char* argv[]) {
    nvtxRangePush("CPU");
    MPI_Init(&argc, &argv);
    setenv("UCX_NET_DEVICES", "ibp3s0:1", 1);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc < 4) {
        if (rank == 0) fprintf(stderr, "Usage: %s <n> <infile> <outfile>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long n = atol(argv[1]);
    const char *infile  = argv[2];
    const char *outfile = argv[3];

    long base = n / size;
    long extra = n % size;
    long local_n = (rank < extra)? (base+1): base;

    int cpu = sched_getcpu();
    int numa_node = numa_node_of_cpu(cpu);
    float *local_data = nullptr;
    if (local_n > 0)
        local_data = (float*)numa_alloc_onnode(sizeof(float) * local_n, numa_node);

    MPI_Offset offset_elems = (rank < extra) ? rank * (base + 1) :
                              (extra * (base + 1) + (rank - extra) * base);
    MPI_Offset file_offset_bytes = offset_elems * sizeof(float);

    // 讀取資料
    MPI_File fh;
    nvtxRangePop(); nvtxRangePush("I/O");
    if (local_n > 0) {
        MPI_File_open(MPI_COMM_SELF, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_read_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }
    nvtxRangePop(); nvtxRangePush("CPU");

    // 用 boost spreadsort 排序
    if (local_n > 0)
        boost::sort::spreadsort::spreadsort(local_data, local_data + local_n);

    // 預先分配最大可能的 neighbor buffer 和 merge buffer
    long max_neighbor_n = base + 1;
    float *neighbor_data = nullptr;
    float *merge_buf = nullptr;
    if (max_neighbor_n > 0)
        neighbor_data = (float*)numa_alloc_onnode(sizeof(float) * max_neighbor_n, numa_node);
    if (local_n > 0)
        merge_buf = (float*)numa_alloc_onnode(sizeof(float) * local_n, numa_node);

    if(local_n == 0){
        MPI_Barrier(MPI_COMM_WORLD);
        if (local_data) numa_free(local_data, sizeof(float) * local_n);
        if (neighbor_data) numa_free(neighbor_data, sizeof(float) * max_neighbor_n);
        if (merge_buf) numa_free(merge_buf, sizeof(float) * local_n);
        MPI_Finalize();
        return 0;
    }

    for(int phase = 0; phase < size+1; phase++){
        bool i_am_left = (phase % 2 == 0) ? rank % 2 == 0 : rank % 2 == 1;
        int peer_rank = i_am_left ? rank + 1 : rank - 1;

        // 檢查 peer 是否存在
        if ((i_am_left && peer_rank >= size) || (!i_am_left && peer_rank < 0))
            continue;

        long neighbor_n = (peer_rank < extra) ? (base + 1) : base;
        if (neighbor_n <= 0) continue;

        // 先傳邊界值 (single float)
        float my_boundary = i_am_left ? local_data[local_n-1] : local_data[0];
        float neighbor_boundary = 0.0f;

        nvtxRangePop(); nvtxRangePush("Comm");
        MPI_Sendrecv(&my_boundary, 1, MPI_FLOAT, peer_rank, 100,
                     &neighbor_boundary, 1, MPI_FLOAT, peer_rank, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nvtxRangePop(); nvtxRangePush("CPU");

        bool need_exchange = i_am_left ? (my_boundary > neighbor_boundary)
                                       : (my_boundary < neighbor_boundary);

        if (!need_exchange)
            continue;

        // 計算要傳送的元素數量 (利用已排序的 local_data)
        int send_count = 0;
        if (i_am_left) {
            // 左側需要傳送所有 > neighbor_boundary 的尾端元素
            // 使用從尾端開始的 galloping（倍增）搜尋，再在縮小後的區間做二分，達到 O(log k)
            // 前提：need_exchange 為真，保證 local_data[local_n-1] > neighbor_boundary（至少會傳 1 個）
            long step = 1;
            long prev_idx = local_n - 1; // 已知這個位置的值 > neighbor_boundary
            long idx = local_n - step;
            while (idx >= 0 && local_data[idx] > neighbor_boundary) {
                prev_idx = idx;
                step <<= 1;
                idx = local_n - step;
            }
            long lo = (idx < 0) ? -1 : idx;     // data[lo] <= boundary（或 lo = -1 作為哨兵）
            long hi = prev_idx;                 // data[hi] > boundary
            // 在 (lo, hi] 內找第一個 > boundary 的位置
            long left = lo + 1;
            long right = hi;
            while (left < right) {
                long mid = left + ((right - left) >> 1);
                if (local_data[mid] > neighbor_boundary) right = mid;
                else left = mid + 1;
            }
            long first_gt = left; // 第一個 > boundary 的索引
            send_count = static_cast<int>(local_n - first_gt);
        } else {
            // 右側需要傳送所有 < neighbor_boundary 的前端元素
            // 從頭開始 galloping（倍增）搜尋找到第一個 >= boundary 的上界，再二分精確化
            long step = 1;
            long prev_idx = 0; // 這裡維持最後一個確認 < boundary 的索引
            long idx = step - 1;
            while (idx < local_n && local_data[idx] < neighbor_boundary) {
                prev_idx = idx;
                step <<= 1;
                idx = step - 1;
            }
            long first_ge;
            if (idx >= local_n && local_data[local_n - 1] < neighbor_boundary) {
                // 全部都 < boundary
                first_ge = local_n;
            } else {
                long lo = prev_idx;                            // data[lo] < boundary
                long hi = std::min<long>(idx, local_n - 1);    // data[hi] >= boundary（或位於邊界內）
                long left = lo + 1;
                long right = hi;
                while (left < right) {
                    long mid = left + ((right - left) >> 1);
                    if (local_data[mid] >= neighbor_boundary) right = mid;
                    else left = mid + 1;
                }
                first_ge = left;
            }
            send_count = static_cast<int>(first_ge); // < boundary 的元素個數
        }

        if (send_count == 0) continue; // 其實不會到這裡，但保險

        // 先互換 send_count -> recv_count
        int recv_count = 0;
        nvtxRangePop(); nvtxRangePush("Comm");
        MPI_Sendrecv(&send_count, 1, MPI_INT, peer_rank, 200,
                     &recv_count, 1, MPI_INT, peer_rank, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nvtxRangePop(); nvtxRangePush("CPU");

        if (recv_count == 0) {
            continue; // 對方沒要送任何東西
        }

        // 確保 recv_count 不會超過我們預先配置的 max_neighbor_n
        if (recv_count > max_neighbor_n) {
            // 若超過，重新配置（理論上不會發生，除非 input 分配不一致）
            numa_free(neighbor_data, sizeof(float) * max_neighbor_n);
            max_neighbor_n = recv_count;
            neighbor_data = (float*)numa_alloc_onnode(sizeof(float) * max_neighbor_n, numa_node);
        }

        // MPI 真正交換資料（只傳需要的子區段）
        nvtxRangePop(); nvtxRangePush("Comm");

        if (i_am_left) {
            // left 傳尾部 send_count 個給右邊，接收來自右邊的頭部 recv_count 個
            MPI_Sendrecv(local_data + (local_n - send_count), send_count, MPI_FLOAT, peer_rank, phase % 2,
                         neighbor_data, recv_count, MPI_FLOAT, peer_rank, phase % 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // right 傳頭部 send_count 個給左邊，接收來自左邊的尾部 recv_count 個
            MPI_Sendrecv(local_data, send_count, MPI_FLOAT, peer_rank, phase % 2,
                         neighbor_data, recv_count, MPI_FLOAT, peer_rank, phase % 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        nvtxRangePop(); nvtxRangePush("CPU");

        // 合併資料：注意這裡使用實際接收到的 recv_count（而非 neighbor_n）
        if (i_am_left) {
            // left 保留最小 local_n 個（從 local_data 與 neighbor_data(來自右邊的 head) 合併）
            merge_keep_smallest_inplace(local_data, (size_t)local_n,
                                        neighbor_data, (size_t)recv_count,
                                        merge_buf, (size_t)local_n);
        } else {
            // right 保留最大 local_n 個（從 local_data 與 neighbor_data(來自左邊的 tail) 合併）
            merge_keep_largest_inplace(local_data, (size_t)local_n,
                                       neighbor_data, (size_t)recv_count,
                                       merge_buf, (size_t)local_n);
        }
        // swap 回 local_data
        std::swap(local_data, merge_buf);
    }

    nvtxRangePop(); nvtxRangePush("IO");
    MPI_File_open(MPI_COMM_SELF, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    nvtxRangePop();

    nvtxRangePush("Comm");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop();

    if (local_data) numa_free(local_data, sizeof(float) * local_n);
    if (neighbor_data) numa_free(neighbor_data, sizeof(float) * max_neighbor_n);
    if (merge_buf) numa_free(merge_buf, sizeof(float) * local_n);
    MPI_Finalize();
    return 0;
}