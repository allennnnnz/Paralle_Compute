#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
int n, m;
static int Dist[5000][5000];  // ⚠️ 注意記憶體上限，不能真的放 50010×50010

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Dist[i][j] = (i == j ? 0 : INF);

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "wb");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ============================================
// 平行化 Block Floyd-Warshall
// ============================================
void cal(int B, int Round, int block_start_x, int block_start_y,
         int block_width, int block_height) {

    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    // block 平行化，將外層的 (b_i, b_j) 二維 block 迴圈攤平成一個迭代空間
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {

            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {

                int block_internal_start_x = b_i * B;
                int block_internal_end_x   = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y   = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int newDist = Dist[i][k] + Dist[k][j];
                        if (newDist < Dist[i][j])
                            Dist[i][j] = newDist;
                    }
                }
            }
        }
    }
}

void block_FW(int B) {
    int round = ceil_div(n, B);
    for (int r = 0; r < round; ++r) {
        printf("Round %d / %d\n", r + 1, round);
        fflush(stdout);

        // Phase 1
        cal(B, r, r, r, 1, 1);

        // Phase 2
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        // Phase 3
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    input(argv[1]);
    int B = 512;

    double start = omp_get_wtime();
    block_FW(B);
    double end = omp_get_wtime();

    printf("Time = %.3f sec\n", end - start);
    output(argv[2]);
    return 0;
}