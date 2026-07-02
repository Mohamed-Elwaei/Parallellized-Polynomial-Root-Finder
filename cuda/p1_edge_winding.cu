// ===========================================================================
// polyroots CUDA port -- Milestone P1: edge-shared winding (the ~2x version)
// ---------------------------------------------------------------------------
// Each UNIQUE grid edge is evaluated exactly once (gather, no atomics), then
// each square's count is assembled from its four cached edge phases:
//
//   count(r,c) = round( ( H[r][c] + V[r][c+1] - H[r+1][c] - V[r][c] ) / 2pi )
//                        \_bottom_/ \_right__/  \__top___/   \_left_/
//
// Canonical directions: horizontal edges L->R, vertical edges Down->Up; the CCW
// square sum gives the +/- signs above. y increases UPWARD (so bottom = H[r]).
// Each edge stores a REAL phase change; rounding happens only per-square.
//
// Kaggle:
//   %%writefile polyroots.cuh        (paste cuda/polyroots.cuh)
//   %%writefile p1_edge_winding.cu   (paste this)
//   !nvcc -O2 -arch=sm_75 -lineinfo p1_edge_winding.cu -o p1 && ./p1
//   !compute-sanitizer ./p1
// ===========================================================================
#include "polyroots.cuh"

// One thread per vertical edge: col in [0,N], r in [0,N).  Direction Down->Up.
__global__ void kVert(const cmplx* coeff, int nc, double ox, double oy, double h,
                      int N, int S, double* V) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col > N || r >= N) return;
    cmplx A(gx(ox, h, N, col), gy(oy, h, N, r));
    cmplx B(gx(ox, h, N, col), gy(oy, h, N, r + 1));
    V[r * (N + 1) + col] = edge_phase(coeff, nc, A, B, S);
}

// One thread per horizontal edge: col in [0,N), r in [0,N].  Direction Left->Right.
__global__ void kHorz(const cmplx* coeff, int nc, double ox, double oy, double h,
                      int N, int S, double* H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || r > N) return;
    cmplx A(gx(ox, h, N, col),     gy(oy, h, N, r));
    cmplx B(gx(ox, h, N, col + 1), gy(oy, h, N, r));
    H[r * N + col] = edge_phase(coeff, nc, A, B, S);
}

// One thread per square: assemble the four cached edges (CCW) and round.
__global__ void kAssemble(const double* V, const double* H, int N, int* counts) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= N || r >= N) return;
    double d = H[r * N + c]               // bottom  (L->R)
             + V[r * (N + 1) + (c + 1)]   // right   (D->U)
             - H[(r + 1) * N + c]         // top     (R->L)
             - V[r * (N + 1) + c];        // left    (U->D)
    double w = d / (2 * kPI);
    counts[r * N + c] = (int)(w >= 0 ? w + 0.5 : w - 0.5);
}

int main() {
    // z^4 - 1, roots at 1, -1, i, -i.
    const int ncoeff = 5;
    cmplx h_coeff[ncoeff] = { cmplx(-1,0), cmplx(0,0), cmplx(0,0), cmplx(0,0), cmplx(1,0) };
    const int degree = ncoeff - 1;

    const int    N  = 16;
    const double L  = 2.0;
    const double h  = L / N;
    const double ox = 0.013, oy = 0.007;
    const int    S  = 64;

    const int nV = N * (N + 1);
    const int nH = (N + 1) * N;
    const int nC = N * N;

    cmplx*  d_coeff = nullptr;
    double* d_V = nullptr; double* d_H = nullptr;
    int*    d_counts = nullptr;
    CK(cudaMalloc(&d_coeff,  ncoeff * sizeof(cmplx)));
    CK(cudaMalloc(&d_V, nV * sizeof(double)));
    CK(cudaMalloc(&d_H, nH * sizeof(double)));
    CK(cudaMalloc(&d_counts, nC * sizeof(int)));
    CK(cudaMemcpy(d_coeff, h_coeff, ncoeff * sizeof(cmplx), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 gV((N + 1 + 15) / 16, (N + 15) / 16);
    dim3 gH((N + 15) / 16, (N + 1 + 15) / 16);
    dim3 gC((N + 15) / 16, (N + 15) / 16);

    kVert<<<gV, block>>>(d_coeff, ncoeff, ox, oy, h, N, S, d_V);
    kHorz<<<gH, block>>>(d_coeff, ncoeff, ox, oy, h, N, S, d_H);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    kAssemble<<<gC, block>>>(d_V, d_H, N, d_counts);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    int* h_counts = (int*)std::malloc(nC * sizeof(int));
    CK(cudaMemcpy(h_counts, d_counts, nC * sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("Polynomial z^4 - 1  (degree %d, roots at 1, -1, i, -i)\n", degree);
    std::printf("Edge-shared winding on %dx%d grid, %d samples/edge\n\n", N, N, S);

    int total = 0, mismatch = 0;
    std::printf("Squares with a nonzero count:\n");
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            int cnt = h_counts[r * N + c];
            total += cnt;
            cmplx ctr(ox + (2 * c + 1 - N) * h, oy + (2 * r + 1 - N) * h);
            int p0 = winding_count(h_coeff, ncoeff, ctr, h, S);  // cross-check vs P0
            if (cnt != p0) ++mismatch;
            if (cnt)
                std::printf("  sq(r=%2d,c=%2d) center=%+.3f%+.3fi  count=%d (P0=%d)\n",
                            r, c, ctr.real(), ctr.imag(), cnt, p0);
        }
    std::printf("\nTotal = %d   (expected degree = %d)        -> %s\n",
                total, degree, total == degree ? "PASS" : "FAIL");
    std::printf("Edge method vs P0 per-cell: %d mismatches  -> %s\n",
                mismatch, mismatch == 0 ? "PASS" : "FAIL");

    std::free(h_counts);
    CK(cudaFree(d_coeff)); CK(cudaFree(d_V)); CK(cudaFree(d_H)); CK(cudaFree(d_counts));
    return 0;
}
