// ===========================================================================
// polyroots CUDA -- scatter + atomics winding (your design)
// ---------------------------------------------------------------------------
// An ALTERNATIVE to p1_edge_winding.cu (gather): instead of each square reading
// its four edges, each EDGE thread computes its phase once and SCATTERS it into
// the two squares it borders, via atomicAdd. Each unique edge is still computed
// exactly once (keeps the ~2x), but assembly is fused into the scatter -- no
// separate H/V tables.
//
// M[r][c] is a DOUBLE phase accumulator (an edge contributes a real fraction of
// 2pi, never an integer). Rounding to a count happens only after all edges have
// scattered. Sign rule (CCW, y-up, canonical edge dirs H:L->R, V:D->U):
//   vertical   edge (r,col): + to square (r,col-1) [its right], - to (r,col) [its left]
//   horizontal edge (r,col): + to square (r,col)   [its bottom], - to (r-1,col) [its top]
// Boundary edges touch only one square, so they are added/subtracted once.
//
// Reproducibility: atomicAdd order is nondeterministic and FP add is not
// associative, so M (and thus a winding sitting exactly on x.5) can wobble in
// the last bits run-to-run. That is expected; this module exists to have the
// scatter form available -- prefer the gather version (P1) when you want
// determinism. atomicAdd(double) needs compute capability >= 6.0 (T4 = 7.5).
//
// Kaggle:
//   %%writefile polyroots.cuh        (paste cuda/polyroots.cuh)
//   %%writefile scatter_winding.cu   (paste this)
//   !nvcc -O2 -arch=sm_75 -lineinfo scatter_winding.cu -o sc && ./sc
//   !compute-sanitizer ./sc
// ===========================================================================
#include "polyroots.cuh"

// One thread per vertical edge: col in [0,N], r in [0,N).  Direction Down->Up.
__global__ void kScatterV(const cmplx* coeff, int nc, double ox, double oy,
                          double h, int N, int S, double* M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col > N || r >= N) return;
    double ph = edge_phase(coeff, nc, cmplx(gx(ox, h, N, col), gy(oy, h, N, r)),
                                      cmplx(gx(ox, h, N, col), gy(oy, h, N, r + 1)), S);
    if (col >= 1) atomicAdd(&M[r * N + (col - 1)],  ph);   // right edge of (r,col-1)
    if (col <  N) atomicAdd(&M[r * N +  col     ], -ph);   // left  edge of (r,col)
}

// One thread per horizontal edge: col in [0,N), r in [0,N].  Direction Left->Right.
__global__ void kScatterH(const cmplx* coeff, int nc, double ox, double oy,
                          double h, int N, int S, double* M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || r > N) return;
    double ph = edge_phase(coeff, nc, cmplx(gx(ox, h, N, col),     gy(oy, h, N, r)),
                                      cmplx(gx(ox, h, N, col + 1), gy(oy, h, N, r)), S);
    if (r <  N) atomicAdd(&M[ r      * N + col],  ph);     // bottom edge of (r,col)
    if (r >= 1) atomicAdd(&M[(r - 1) * N + col], -ph);     // top    edge of (r-1,col)
}

// One thread per square: round the accumulated phase to an integer count.
__global__ void kRound(const double* M, int N, int* counts) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= N || r >= N) return;
    double w = M[r * N + c] / (2 * kPI);
    counts[r * N + c] = (int)(w >= 0 ? w + 0.5 : w - 0.5);
}

int main() {
    // z^4 - 1, roots at 1, -1, i, -i.
    const int ncoeff = 5;
    cmplx h_coeff[ncoeff] = { cmplx(-1,0), cmplx(0,0), cmplx(0,0), cmplx(0,0), cmplx(1,0) };
    const int degree = ncoeff - 1;

    const int    N  = 16;
    const double L  = 2.0, h = L / N, ox = 0.013, oy = 0.007;
    const int    S  = 64;
    const int    nC = N * N;

    cmplx*  d_coeff = nullptr;
    double* d_M = nullptr;
    int*    d_counts = nullptr;
    CK(cudaMalloc(&d_coeff,  ncoeff * sizeof(cmplx)));
    CK(cudaMalloc(&d_M,      nC * sizeof(double)));
    CK(cudaMalloc(&d_counts, nC * sizeof(int)));
    CK(cudaMemcpy(d_coeff, h_coeff, ncoeff * sizeof(cmplx), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_M, 0, nC * sizeof(double)));            // M starts at zero

    dim3 block(16, 16);
    dim3 gV((N + 1 + 15) / 16, (N + 15) / 16);  // vertical edges: cols 0..N, rows 0..N-1
    dim3 gH((N + 15) / 16, (N + 1 + 15) / 16);  // horizontal edges: cols 0..N-1, rows 0..N
    dim3 gC((N + 15) / 16, (N + 15) / 16);

    // Same default stream -> these run in order; the only contention is the
    // atomicAdds into shared M entries, which is exactly what atomics handle.
    kScatterV<<<gV, block>>>(d_coeff, ncoeff, ox, oy, h, N, S, d_M);
    kScatterH<<<gH, block>>>(d_coeff, ncoeff, ox, oy, h, N, S, d_M);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    kRound<<<gC, block>>>(d_M, N, d_counts);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    int* h_counts = (int*)std::malloc(nC * sizeof(int));
    CK(cudaMemcpy(h_counts, d_counts, nC * sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("Polynomial z^4 - 1  (degree %d)  --  scatter+atomics winding\n", degree);
    std::printf("Grid %dx%d, %d samples/edge\n\n", N, N, S);

    int total = 0, mismatch = 0;
    std::printf("Squares with a nonzero count:\n");
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            int cnt = h_counts[r * N + c];
            total += cnt;
            cmplx ctr(ox + (2 * c + 1 - N) * h, oy + (2 * r + 1 - N) * h);
            int p0 = winding_count(h_coeff, ncoeff, ctr, h, S);   // cross-check vs P0
            if (cnt != p0) ++mismatch;
            if (cnt)
                std::printf("  sq(r=%2d,c=%2d) center=%+.3f%+.3fi  count=%d (P0=%d)\n",
                            r, c, ctr.real(), ctr.imag(), cnt, p0);
        }
    std::printf("\nTotal = %d   (expected degree = %d)        -> %s\n",
                total, degree, total == degree ? "PASS" : "FAIL");
    std::printf("Scatter vs P0 per-cell: %d mismatches      -> %s\n",
                mismatch, mismatch == 0 ? "PASS" : "FAIL");

    std::free(h_counts);
    CK(cudaFree(d_coeff)); CK(cudaFree(d_M)); CK(cudaFree(d_counts));
    return 0;
}
