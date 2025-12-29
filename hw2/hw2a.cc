// pthread + simd (3-group per step, cyclic row distribution, atomic fetch)


/*
1.	建立任務（Task）
	•	用 build_row_tasks_cyclic() 或其他方式，把每列的任務 (RowTask) 建好。
	•	每個任務只包含 j_global（這列的索引）。
2.	建立任務佇列（TaskQueue）
	•	將所有任務放進 TaskQueue.tasks。
	•	初始化 TaskQueue.ntasks 與 TaskQueue.next 原子計數器（初始為 0）。
3.	啟動多個執行緒
	•	每個 thread 都呼叫 worker_main()。
	•	依需求可用 maybe_pin_this_thread() 將 thread 綁定到特定 CPU（Linux 專用）。
4.	動態搶任務
	•	每個 thread 在迴圈裡用：
    start_idx = q->next.fetch_add(batch_size);
5.	處理任務
	•	每個 thread 依序處理自己搶到的任務：
    compute_row(ctx, j_global);
6.	結束 thread
	•	當 claim_next_task_batch() 回傳 count = 0，表示所有任務都被搶完。
	•	thread 結束迴圈並 return NULL。
7.	主 thread 等待
	•	主程式用 pthread_join() 等所有 thread 完成。
8.	寫出結果
	•	所有 row 計算完成後，呼叫 write_png() 將 ctx.image 寫成 PNG。
*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <immintrin.h>
#include <atomic>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
#define nvtxRangePush(x) ((void)0)
#define nvtxRangePop() ((void)0)
#endif

const __m128d c1 = _mm_set1_pd(1.0);
const __m128d c2 = _mm_set1_pd(2.0);
const __m128d c0 = _mm_setzero_pd();
const __m128d c4 = _mm_set1_pd(4.0);

/* ===== PNG writer: row-major + 垂直翻轉 ===== */
static void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, (png_uint_32)width, (png_uint_32)height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    size_t row_size = (size_t)3 * (size_t)width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    assert(row);

    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x]; // 垂直翻轉
            png_bytep color = row + (size_t)x * 3u;
            if (p != iters) {
                if (p & 16) {             // 簡單色盤
                    color[0] = 240;
                    color[1] = color[2] = (png_byte)((p % 16) * 16);
                } else {
                    color[0] = (png_byte)((p % 16) * 16);
                    color[1] = 0;
                    color[2] = 0;
                }
            } else {
                color[0] = color[1] = color[2] = 0;
            }
        }
        png_write_row(png_ptr, row);
    }

    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* ===== 迭代步（SIMD / scalar，與 hw2b 一致採用指標參數） ===== */
static inline __m128d iter_cal(__m128d *x, __m128d *y, __m128d *xx, __m128d *yy,
                               const __m128d x0, const __m128d y0) {
    __m128d xy = _mm_mul_pd(*x, *y);
    *x = _mm_add_pd(_mm_sub_pd(*xx, *yy), x0);
    *y = _mm_add_pd(_mm_mul_pd(c2, xy), y0);
    *xx = _mm_mul_pd(*x, *x);
    *yy = _mm_mul_pd(*y, *y);
    return _mm_add_pd(*xx, *yy);
}

static inline double iter_cal_scalar(double *x, double *y, double *xx, double *yy,
                                     const double x0, const double y0) {
    double xy = (*x) * (*y);
    *x = (*xx) - (*yy) + x0;
    *y = 2.0 * xy + y0;
    *xx = (*x) * (*x);
    *yy = (*y) * (*y);
    return (*xx) + (*yy);
}

/* ===== 工作列結構每個任務負責一列 ===== */
typedef struct {
    int j_global; // 全域列索引
} RowTask;

/* ===== 佇列（改用原子操作，無 mutex） ===== */
//每次thread可以來這個queue取一個任務
typedef struct {
    RowTask* tasks;
    int ntasks;
    std::atomic<int> next;  // 改用原子變數
} TaskQueue;

/* ===== 共享上下文 ===== */
typedef struct {
    int iters;
    double left, right, lower, upper;
    int width, height;
    double dxi, dyj;
    int* image; // row-major size width*height
    TaskQueue q;
    int nthreads;
    int pin_threads;
    int ncpus;
    int use_cyclic; // 0=block, 1=cyclic
    int batch_size; // 每次取多少列（減少原子操作頻率）
} Context;

typedef struct {
    Context* ctx;
    int tid;
} ThreadArg;

// 取得可用 CPU 數量
static int get_cpu_count(void) {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) return CPU_COUNT(&set);
    long n = -1;
#ifdef _SC_NPROCESSORS_ONLN
    n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return (n > 0) ? (int)n : 1;
}

// 若啟用則把執行緒綁定到指定 CPU 上（Linux 專用，否則為 no-op）
static void maybe_pin_this_thread(int tid, int ncpus, int enable) {
    if (!enable || ncpus <= 0) return;
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(tid % ncpus, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid; (void)ncpus;
#endif
}

// 從原子計數器中批次索取任務索引（減少原子操作頻率），取玩當前的帶解覺任物指標後，next會自動+1，代表下一個任務索引
static int claim_next_task_batch(TaskQueue* q, int batch_size, int* out_count) {
    int idx = q->next.fetch_add(batch_size, std::memory_order_relaxed);
    if (idx >= q->ntasks) {
        *out_count = 0;
        return idx;
    }
    // 計算實際可取得的任務數（避免超過總數）
    int remaining = q->ntasks - idx;
    *out_count = (remaining < batch_size) ? remaining : batch_size;
    return idx;
}

// 把資料切分成一個個row，共會有height個任務
static RowTask* build_row_tasks_cyclic(int height, int nthreads, int* out_n) {
    // 每個執行緒負責 tid, tid + nthreads, tid + 2*nthreads, ...
    // 總任務數 = height（每列是一個獨立任務）
    RowTask* v = (RowTask*)malloc(sizeof(RowTask) * (size_t)height);
    assert(v);
    for (int j = 0; j < height; ++j) {
        //j_global 就是我現在這個任務是處理哪個row
        v[j].j_global = j;
    }
    *out_n = height;
    return v;
}

// 計算一列內所有像素的 Mandelbrot 值，支援 3 組 SIMD + 尾端 2-pixel + 單像素處理
// 這是每個thread娶到任務後會呼叫的函式
static void compute_row(const Context* ctx, int j_global) {
    const int iters = ctx->iters;
    const int width = ctx->width;
    const double dxi = ctx->dxi;
    const double dyj = ctx->dyj;
    const double left = ctx->left;
    const double lower = ctx->lower;

    double y0 = (double)j_global * dyj + lower;
    __m128d ys = _mm_set1_pd(y0);
    int* rowp = ctx->image + (size_t)j_global * (size_t)width;
    
    // Prefetch 下一列的快取行（改善記憶體延遲）
    if (j_global + 1 < ctx->height) {
        int* next_row = ctx->image + (size_t)(j_global + 1) * (size_t)width;
        __builtin_prefetch(next_row, 1, 0);
    }

    int i = 0;
    // 一次處理 6 個像素（3 組 SIMD，每組 2 像素）
    for (; i + 5 < width; i += 6) {
        __m128d xs0 = _mm_set_pd((double)(i+1) * dxi + left, (double)(i+0) * dxi + left);
        __m128d xs1 = _mm_set_pd((double)(i+3) * dxi + left, (double)(i+2) * dxi + left);
        __m128d xs2 = _mm_set_pd((double)(i+5) * dxi + left, (double)(i+4) * dxi + left);

        __m128d x0v=c0, y0v=c0, xx0=c0, yy0=c0, rep0=c0, len0=c0;
        __m128d x1v=c0, y1v=c0, xx1=c0, yy1=c0, rep1=c0, len1=c0;
        __m128d x2v=c0, y2v=c0, xx2=c0, yy2=c0, rep2=c0, len2=c0;

        __m128d m0 = _mm_castsi128_pd(_mm_set1_epi64x(-1));
        __m128d m1 = _mm_castsi128_pd(_mm_set1_epi64x(-1));
        __m128d m2 = _mm_castsi128_pd(_mm_set1_epi64x(-1));

        int k = 0;
        while (k < iters) {
            int st_any = (_mm_movemask_pd(m0) | _mm_movemask_pd(m1) | _mm_movemask_pd(m2));
            if (st_any == 0) break;
            for (int t = 0; t < 64 && k < iters; ++t, ++k) {
                len0 = iter_cal(&x0v, &y0v, &xx0, &yy0, xs0, ys);
                rep0 = _mm_add_pd(rep0, _mm_and_pd(m0, c1));
                m0   = _mm_cmplt_pd(len0, c4);

                len1 = iter_cal(&x1v, &y1v, &xx1, &yy1, xs1, ys);
                rep1 = _mm_add_pd(rep1, _mm_and_pd(m1, c1));
                m1   = _mm_cmplt_pd(len1, c4);

                len2 = iter_cal(&x2v, &y2v, &xx2, &yy2, xs2, ys);
                rep2 = _mm_add_pd(rep2, _mm_and_pd(m2, c1));
                m2   = _mm_cmplt_pd(len2, c4);
            }
        }
        __m128d itv = _mm_set1_pd((double)iters);
        rep0 = _mm_min_pd(rep0, itv);
        rep1 = _mm_min_pd(rep1, itv);
        rep2 = _mm_min_pd(rep2, itv);

        double r0[2], r1[2], r2[2];
        _mm_storeu_pd(r0, rep0);
        _mm_storeu_pd(r1, rep1);
        _mm_storeu_pd(r2, rep2);

        rowp[i+0] = (int)r0[0];
        rowp[i+1] = (int)r0[1];
        rowp[i+2] = (int)r1[0];
        rowp[i+3] = (int)r1[1];
        rowp[i+4] = (int)r2[0];
        rowp[i+5] = (int)r2[1];
    }

    // 尾端 2 像素
    for (; i + 1 < width; i += 2) {
        __m128d xs = _mm_set_pd((double)(i+1) * dxi + left, (double)(i+0) * dxi + left);
        __m128d x=c0,y=c0,xx=c0,yy=c0,rep=c0,len=c0;
        __m128d m = _mm_castsi128_pd(_mm_set1_epi64x(-1));
        int k = 0;
        while (k < iters) {
            int st = _mm_movemask_pd(m);
            if (st == 0) break;
            for (int t = 0; t < 64 && k < iters; ++t, ++k) {
                len = iter_cal(&x, &y, &xx, &yy, xs, ys);
                rep = _mm_add_pd(rep, _mm_and_pd(m, c1));
                m   = _mm_cmplt_pd(len, c4);
            }
        }
        __m128d itv = _mm_set1_pd((double)iters);
        rep = _mm_min_pd(rep, itv);
        double rt[2];
        _mm_storeu_pd(rt, rep);
        rowp[i+0] = (int)rt[0];
        rowp[i+1] = (int)rt[1];
    }

    // 單像素尾巴
    if (i < width) {
        int repeats = 0;
        double x=0.0,y=0.0,xx=0.0,yy=0.0;
        double len = 0.0;
        double x0 = (double)i * dxi + left;
        while (repeats < iters && len < 4.0) {
            len = iter_cal_scalar(&x,&y,&xx,&yy,x0,y0);
            ++repeats;
        }
        rowp[i] = repeats;
    }
}

// 工作執行緒主迴圈：批次取任務並呼叫 compute_row，直到任務耗盡
static void* worker_main(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;
    Context* ctx = ta->ctx;
    maybe_pin_this_thread(ta->tid, ctx->ncpus, ctx->pin_threads);
    
    int batch_size = ctx->batch_size;
    for (;;) {
        int count = 0;
        int start_idx = claim_next_task_batch(&ctx->q, batch_size, &count);
        if (count == 0) break;
        
        // 處理這一批任務
        for (int i = 0; i < count; ++i) {
            int idx = start_idx + i;
            int j_global = ctx->q.tasks[idx].j_global;
            compute_row(ctx, j_global);
        }
    }
    return NULL;
}

// 由環境變數讀取整數值，若不存在或格式錯誤則回傳預設值
static int env_get_int(const char* key, int defv) {
    const char* s = getenv(key);
    if (!s || !*s) return defv;
    char* endp = NULL;
    long v = strtol(s, &endp, 10);
    return (endp && *endp == '\0') ? (int)v : defv;
}

int main(int argc, char** argv) {
    nvtxRangePush("CPU");
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <filename.png> <iters> <left> <right> <lower> <upper> <width> <height>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    const int   iters  = (int)strtol(argv[2], 0, 10);
    const double left  = strtod(argv[3], 0);
    const double right = strtod(argv[4], 0);
    const double lower = strtod(argv[5], 0);
    const double upper = strtod(argv[6], 0);
    const int   width  = (int)strtol(argv[7], 0, 10);
    const int   height = (int)strtol(argv[8], 0, 10);

    Context ctx;
    ctx.iters = iters;
    ctx.left = left; ctx.right = right;
    ctx.lower = lower; ctx.upper = upper;
    ctx.width = width; ctx.height = height;
    ctx.dxi = (right - left) / (double)width;
    ctx.dyj = (upper - lower) / (double)height;

    ctx.ncpus = get_cpu_count();
    ctx.nthreads = env_get_int("NTHREADS", ctx.ncpus);
    ctx.pin_threads = env_get_int("PIN_THREADS", 0) ? 1 : 0;
    
    // 批次大小：每次取多少列（預設 8，可用環境變數調整）
    ctx.batch_size = env_get_int("BATCH_SIZE", 8);

    // 分配策略：預設 cyclic 打散（與 hw2b 一致）
    const char* dist_env = getenv("PTHREAD_DIST");
    ctx.use_cyclic = 1;
    if (dist_env && *dist_env) {
        if (strcasecmp(dist_env, "block") == 0) ctx.use_cyclic = 0;
        else if (strcasecmp(dist_env, "cyclic") == 0) ctx.use_cyclic = 1;
    }

    int* image = (int*)malloc((size_t)width * (size_t)height * sizeof(int));
    assert(image);
    ctx.image = image;

    // 建立列任務佇列（cyclic 模式：所有列按順序排列，執行緒動態取）
    int ntasks = 0;
    RowTask* tasks = build_row_tasks_cyclic(height, ctx.nthreads, &ntasks);
    ctx.q.tasks = tasks;
    ctx.q.ntasks = ntasks;
    // C++ atomic 不需要 atomic_init，使用建構子初始化
    ctx.q.next.store(0, std::memory_order_relaxed);

    // 啟動工作執行緒
    pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)ctx.nthreads);
    ThreadArg* ta = (ThreadArg*)malloc(sizeof(ThreadArg) * (size_t)ctx.nthreads);
    assert(th && ta);

    for (int t = 0; t < ctx.nthreads; ++t) {
        ta[t].ctx = &ctx;
        ta[t].tid = t;
        int rc = pthread_create(&th[t], NULL, worker_main, &ta[t]);
        assert(rc == 0);
    }
    for (int t = 0; t < ctx.nthreads; ++t) {
        pthread_join(th[t], NULL);
    }

    free(ta);
    free(th);
    free(tasks);

    nvtxRangePop(); nvtxRangePush("I/O");
    write_png(filename, iters, width, height, image);
    nvtxRangePop(); nvtxRangePush("CPU");
    free(image);
    return 0;
}
