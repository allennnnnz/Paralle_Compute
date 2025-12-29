// mpi + openmp + simd (3-group per step)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include <nvtx3/nvToolsExt.h>

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

/* ===== 迭代步（SIMD / scalar） ===== */
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

int main(int argc, char** argv) {
    /* ===== MPI init ===== */
    nvtxRangePush("CPU");
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ===== 參數解析 ===== */
    if (argc != 9) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: %s <filename.png> <iters> <left> <right> <lower> <upper> <width> <height>\n",
                argv[0]);
        }
        nvtxRangePop(); nvtxRangePush("Comm");
        MPI_Abort(MPI_COMM_WORLD, 1);
        nvtxRangePop(); nvtxRangePush("CPU");
    }
    const char* filename = argv[1];
    const int   iters  = (int)strtol(argv[2], 0, 10);
    const double left  = strtod(argv[3], 0);
    const double right = strtod(argv[4], 0);
    const double lower = strtod(argv[5], 0);
    const double upper = strtod(argv[6], 0);
    const int   width  = (int)strtol(argv[7], 0, 10);
    const int   height = (int)strtol(argv[8], 0, 10);

    /* ===== 分配策略：block / cyclic ===== */
    const char* dist_env = getenv("MPI_DIST");
    int use_cyclic = 1; // 預設 cyclic 打散
    if (dist_env && *dist_env) {
        if (strcasecmp(dist_env, "block") == 0) use_cyclic = 0;
        else if (strcasecmp(dist_env, "cyclic") == 0) use_cyclic = 1;
    }

    const int rbase   = height / size;
    const int rextra  = height % size;

    int my_rows = 0;
    int start_row = 0;
    if (!use_cyclic) {
        my_rows   = rbase + (rank < rextra ? 1 : 0);
        start_row = rank * rbase + (rank < rextra ? rank : rextra);
    } else {
        for (int j = rank; j < height; j += size) ++my_rows;
        start_row = -1; // 非連續
    }
    const int local_n = my_rows * width;

    int* image_part = (int*)malloc((size_t)local_n * sizeof(int));
    assert(image_part);

    /* ===== 座標步距 ===== */
    const double dxi = (right - left) / (double)width;
    const double dyj = (upper - lower) / (double)height;

    /* ===== 節點內並行：外層列動態分派；內層 3 組 SIMD（6 像素/步） ===== */
    omp_set_nested(0);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j_local = 0; j_local < my_rows; ++j_local) {
        int j_global = (!use_cyclic) ? (start_row + j_local) : (rank + j_local * size);
        double y0 = (double)j_global * dyj + lower;
        __m128d ys = _mm_set1_pd(y0);

        int* rowp = image_part + (size_t)j_local * (size_t)width;

        int i = 0;
        for (; i + 5 < width; i += 6) {
            // 3 組 x 座標（每組 2 像素，lane1=高位、lane0=低位）
            __m128d xs0 = _mm_set_pd((double)(i+1) * dxi + left,
                                     (double)(i+0) * dxi + left);
            __m128d xs1 = _mm_set_pd((double)(i+3) * dxi + left,
                                     (double)(i+2) * dxi + left);
            __m128d xs2 = _mm_set_pd((double)(i+5) * dxi + left,
                                     (double)(i+4) * dxi + left);

            // 狀態：3 組
            __m128d x0v=c0, y0v=c0, xx0=c0, yy0=c0, rep0=c0, len0=c0;
            __m128d x1v=c0, y1v=c0, xx1=c0, yy1=c0, rep1=c0, len1=c0;
            __m128d x2v=c0, y2v=c0, xx2=c0, yy2=c0, rep2=c0, len2=c0;

            __m128d m0 = _mm_castsi128_pd(_mm_set1_epi64x(-1)); // all-true
            __m128d m1 = _mm_castsi128_pd(_mm_set1_epi64x(-1));
            __m128d m2 = _mm_castsi128_pd(_mm_set1_epi64x(-1));

            int k = 0;
            while (k < iters) {
                int st_any = (_mm_movemask_pd(m0) |
                              _mm_movemask_pd(m1) |
                              _mm_movemask_pd(m2));
                if (st_any == 0) break;

                for (int t = 0; t < 64 && k < iters; ++t, ++k) {
                    // 組0
                    len0 = iter_cal(&x0v, &y0v, &xx0, &yy0, xs0, ys);
                    rep0 = _mm_add_pd(rep0, _mm_and_pd(m0, c1));
                    m0   = _mm_cmplt_pd(len0, c4);
                    // 組1
                    len1 = iter_cal(&x1v, &y1v, &xx1, &yy1, xs1, ys);
                    rep1 = _mm_add_pd(rep1, _mm_and_pd(m1, c1));
                    m1   = _mm_cmplt_pd(len1, c4);
                    // 組2
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
            _mm_storeu_pd(r0, rep0); // r0[0] -> i,   r0[1] -> i+1
            _mm_storeu_pd(r1, rep1); // r1[0] -> i+2, r1[1] -> i+3
            _mm_storeu_pd(r2, rep2); // r2[0] -> i+4, r2[1] -> i+5

            rowp[i+0] = (int)r0[0];
            rowp[i+1] = (int)r0[1];
            rowp[i+2] = (int)r1[0];
            rowp[i+3] = (int)r1[1];
            rowp[i+4] = (int)r2[0];
            rowp[i+5] = (int)r2[1];
        }

        // 2-像素尾巴（盡量吃到只剩 0 或 1 個）
        for (; i + 1 < width; i += 2) {
            __m128d xs = _mm_set_pd((double)(i+1) * dxi + left,
                                    (double)(i+0) * dxi + left);

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

    /* ===== 收合到 root ===== */
    int total_elems = width * height;
    if (!use_cyclic) {
        // block：Gatherv 串接 row-major
        int* recvcounts = NULL;
        int* displs = NULL;
        int* image_full = NULL;
        if (rank == 0) {
            recvcounts = (int*)malloc((size_t)size * sizeof(int));
            displs     = (int*)malloc((size_t)size * sizeof(int));
            assert(recvcounts && displs);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                int rows_r = rbase + (r < rextra ? 1 : 0);
                recvcounts[r] = rows_r * width;
                displs[r]     = offset;
                offset += recvcounts[r];
            }
            image_full = (int*)malloc((size_t)total_elems * sizeof(int));
            assert(image_full);
        }
        nvtxRangePop(); nvtxRangePush("Comm");
        MPI_Gatherv(
            image_part, local_n, MPI_INT,
            image_full, recvcounts, displs, MPI_INT,
            0, MPI_COMM_WORLD
        );
        nvtxRangePop(); nvtxRangePush("CPU");
        if (rank == 0) {
            nvtxRangePop(); nvtxRangePush("IO");
            write_png(filename, iters, width, height, image_full);
            nvtxRangePop(); nvtxRangePush("CPU");
            free(image_full);
            free(recvcounts);
            free(displs);
        }
    } else {
        // cyclic：派生型別，Irecv 直寫交錯列
        MPI_Datatype row_t; // 一列
        MPI_Type_contiguous(width, MPI_INT, &row_t);
        MPI_Type_commit(&row_t);

        MPI_Request send_req = MPI_REQUEST_NULL;
        if (my_rows > 0) {
            MPI_Isend(image_part, my_rows, row_t, 0, 0, MPI_COMM_WORLD, &send_req);
        }

        int* final_image = NULL;
        MPI_Request* recv_reqs = NULL;
        MPI_Datatype* vec_types = NULL;
        if (rank == 0) {
            final_image = (int*)malloc((size_t)total_elems * sizeof(int));
            assert(final_image);
            recv_reqs = (MPI_Request*)malloc((size_t)size * sizeof(MPI_Request));
            vec_types = (MPI_Datatype*)malloc((size_t)size * sizeof(MPI_Datatype));
            assert(recv_reqs && vec_types);
            for (int r = 0; r < size; ++r) {
                int rows_r = 0;
                for (int j = r; j < height; j += size) ++rows_r;
                if (rows_r == 0) { recv_reqs[r] = MPI_REQUEST_NULL; vec_types[r] = MPI_DATATYPE_NULL; continue; }
                MPI_Aint stride_bytes = (MPI_Aint)size * (MPI_Aint)width * (MPI_Aint)sizeof(int);
                MPI_Datatype vec_t;
                MPI_Type_create_hvector(rows_r, 1, stride_bytes, row_t, &vec_t);
                MPI_Type_commit(&vec_t);
                vec_types[r] = vec_t;
                int* base_ptr = final_image + (size_t)r * (size_t)width;
                MPI_Irecv(base_ptr, 1, vec_t, r, 0, MPI_COMM_WORLD, &recv_reqs[r]);
            }
        }

        nvtxRangePop(); nvtxRangePush("Comm");
        if (rank == 0) {
            MPI_Waitall(size, recv_reqs, MPI_STATUSES_IGNORE);
        }
        if (send_req != MPI_REQUEST_NULL) MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        nvtxRangePop(); nvtxRangePush("CPU");

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                if (vec_types[r] != MPI_DATATYPE_NULL) MPI_Type_free(&vec_types[r]);
            }
            free(vec_types);
            free(recv_reqs);
            nvtxRangePop(); nvtxRangePush("IO");
            write_png(filename, iters, width, height, final_image);
            nvtxRangePop(); nvtxRangePush("CPU");
            free(final_image);
        }
        MPI_Type_free(&row_t);
    }

    free(image_part);
    MPI_Finalize();
    return 0;
}