// ===========================================================================
// polyroots CUDA port -- Milestone P0: grid winding, one thread per cell.
// ---------------------------------------------------------------------------
// Smallest useful kernel: a flat grid of cells, ONE THREAD PER CELL, each
// computing the winding-number root count of its cell. Proves the toolchain,
// the device math, and host==device agreement. Shared math lives in the header.
//
// Kaggle:
//   %%writefile polyroots.cuh         (paste cuda/polyroots.cuh)
//   %%writefile p0_grid_winding.cu    (paste this)
//   !nvcc -O2 -arch=sm_75 -lineinfo p0_grid_winding.cu -o p0 && ./p0
//   !compute-sanitizer ./p0
// ===========================================================================
#include "polyroots.cuh"

// One thread per grid cell. The grid is GX x GY square cells of half-width
// cellHalf, cell (i,j) centred at origin + ((2i+1-GX)*cellHalf,(2j+1-GY)*cellHalf).
// The tiny `origin` offset keeps roots off cell edges (where arg P is undefined).
__global__
void grid_winding(const cmplx* coeff, int ncoeff, cmplx origin, double cellHalf,
                  int GX, int GY, int samplesPerSide, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (i >= GX || j >= GY) return;
    cmplx center = origin + cmplx((2 * i + 1 - GX) * cellHalf,
                                  (2 * j + 1 - GY) * cellHalf);
    counts[j * GX + i] = winding_count(coeff, ncoeff, center, cellHalf, samplesPerSide);
}

int main() {
    // z^4 - 1, roots at 1, -1, i, -i.  Coefficients ascending.
    const int ncoeff = 5;
    cmplx h_coeff[ncoeff] = { cmplx(-1,0), cmplx(0,0), cmplx(0,0), cmplx(0,0), cmplx(1,0) };
    const int degree = ncoeff - 1;

    const int    N        = 16;
    const double L        = 2.0;
    const double cellHalf = L / N;                 // 0.125
    const cmplx  origin   = cmplx(0.013, 0.007);   // anti-grid-line offset
    const int    sps      = 64;
    const int    ncells   = N * N;

    cmplx* d_coeff = nullptr; int* d_counts = nullptr;
    CK(cudaMalloc(&d_coeff,  ncoeff * sizeof(cmplx)));
    CK(cudaMalloc(&d_counts, ncells * sizeof(int)));
    CK(cudaMemcpy(d_coeff, h_coeff, ncoeff * sizeof(cmplx), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    grid_winding<<<grid, block>>>(d_coeff, ncoeff, origin, cellHalf, N, N, sps, d_counts);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    int* h_counts = (int*)std::malloc(ncells * sizeof(int));
    CK(cudaMemcpy(h_counts, d_counts, ncells * sizeof(int), cudaMemcpyDeviceToHost));

    int total = 0;
    std::printf("Polynomial z^4 - 1  (degree %d, roots at 1, -1, i, -i)\n", degree);
    std::printf("Grid %dx%d over [-%.1f,%.1f]^2, cellHalf=%.4f, %d samples/side\n\n",
                N, N, L, L, cellHalf, sps);
    std::printf("Cells with a nonzero count:\n");
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            int c = h_counts[j * N + i];
            total += c;
            if (c != 0) {
                cmplx ctr = origin + cmplx((2 * i + 1 - N) * cellHalf,
                                           (2 * j + 1 - N) * cellHalf);
                std::printf("  cell(%2d,%2d) center=%+.3f%+.3fi  count=%d\n",
                            i, j, ctr.real(), ctr.imag(), c);
            }
        }
    std::printf("\nTotal winding over grid = %d   (expected = degree = %d)  -> %s\n",
                total, degree, total == degree ? "PASS" : "FAIL");

    int mismatches = 0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            cmplx ctr = origin + cmplx((2 * i + 1 - N) * cellHalf,
                                       (2 * j + 1 - N) * cellHalf);
            int cpu = winding_count(h_coeff, ncoeff, ctr, cellHalf, sps);
            if (cpu != h_counts[j * N + i]) ++mismatches;
        }
    std::printf("Host vs device: %d mismatching cells  -> %s\n",
                mismatches, mismatches == 0 ? "PASS" : "FAIL");

    std::free(h_counts);
    CK(cudaFree(d_coeff)); CK(cudaFree(d_counts));
    return 0;
}
