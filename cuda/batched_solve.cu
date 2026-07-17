// ===========================================================================
// polyroots CUDA -- batched solver (throughput)
// ---------------------------------------------------------------------------
// Solves a whole BATCH of polynomials at once. The work-list holds
// (polynomial, cell) tasks across ALL polynomials, so one kernel launch fills
// the GPU with thousands of independent cells -- the framing where the GPU's
// parallelism beats a serial CPU/NumPy loop on THROUGHPUT (polynomials/sec).
//
// v1 scope: all polynomials the same degree (dense N x (deg+1) coefficient
// matrix), per-polynomial root bound, host-driven BFS (the work-list lives on
// the host; moving it on-device with CUB is the v2 win). One block per
// (poly,cell) task; 64 threads split that cell's 8x8 sub-grid (naive winding).
//
// I/O:  ./batched_solve <N> <degree>    then N*(degree+1) "re im" pairs on stdin
//       (each polynomial's coeffs DESCENDING, polynomials back to back).
// Out:  SOLVE_MS <ms>   THROUGHPUT <polys/sec>   MAXRESIDUAL <r>   ROOTS <total>
//
// Kaggle:
//   %%writefile polyroots.cuh     (paste cuda/polyroots.cuh)
//   %%writefile batched_solve.cu  (paste this)
//   !nvcc -O2 -arch=sm_75 batched_solve.cu -o batched
// ===========================================================================
#include "polyroots.cuh"
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

constexpr int K    = 8;          // KxK sub-grid per cell
constexpr int NSUB = K * K;
constexpr int BLK  = 64;

struct BatchCell { int poly; double cx, cy, half; };   // a (polynomial, cell) task

// ---- device-side Cauchy/Fujiwara bound (used by the parallel setup kernel) --
__device__ inline double d_cauchy(const cmplx* c, int nc) {
    int n = nc - 1; double cn = thrust::abs(c[n]), m = 0;
    for (int k = 0; k < n; ++k) { double v = thrust::abs(c[k]) / cn; m = v > m ? v : m; }
    return 1.0 + m;
}
__device__ inline double d_fujiwara(const cmplx* c, int nc) {
    int n = nc - 1; double cn = thrust::abs(c[n]), m = 0;
    for (int k = 1; k <= n; ++k) { double r = thrust::abs(c[n-k]) / cn; if (k == n) r *= 0.5;
        double v = pow(r, 1.0 / k); m = v > m ? v : m; }
    return 2.0 * m;
}
__device__ inline double d_bound(const cmplx* c, int nc) {
    double a = d_cauchy(c, nc), b = d_fujiwara(c, nc); return a < b ? a : b;
}

// ---- PARALLEL setup: one thread per polynomial -----------------------------
// Computes each polynomial's derivative + root bound, and writes its ONE
// initial task directly to frontier[p] (no atomics: exactly one task per poly).
__global__ void setup(const cmplx* coeffs, int N, int nc, cmplx* deriv,
                      double* bounds, BatchCell* frontier) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    const cmplx* C = coeffs + (size_t)p * nc;
    cmplx* Dp = deriv + (size_t)p * (nc - 1);
    for (int k = 1; k < nc; ++k) Dp[k-1] = C[k] * (double)k;   // derivative
    double R = d_bound(C, nc);
    bounds[p] = R;
    double off = R * 0.05;
    frontier[p] = { p, off * 0.4142, off * 0.3141, R * 1.05 + off };
}

// ---- winding: one block per (poly,cell) task, 64 threads over the sub-grid --
__global__ void batched_winding(const BatchCell* frontier, const cmplx* coeffs,
                                int nc, int sps, int* counts) {
    int b = blockIdx.x;
    BatchCell tk = frontier[b];
    const cmplx* C = coeffs + (size_t)tk.poly * nc;           // <-- this poly's row
    double h = tk.half / K;
    for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
        int i = t % K, j = t / K;
        cmplx ctr(tk.cx - tk.half + (2*i+1)*h, tk.cy - tk.half + (2*j+1)*h);
        counts[(size_t)b * NSUB + t] = winding_count(C, nc, ctr, h, sps);
    }
}

// ---- Newton: one thread per isolated cell ----------------------------------
__global__ void batched_newton(const BatchCell* iso, int m, const cmplx* coeffs,
                               const cmplx* deriv, int nc, cmplx* out, int* ok, int* polyOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    BatchCell tk = iso[i];
    const cmplx* C  = coeffs + (size_t)tk.poly * nc;
    const cmplx* Dp = deriv  + (size_t)tk.poly * (nc - 1);
    cmplx c0(tk.cx, tk.cy);
    cmplx z = newton_polish(C, nc, Dp, nc - 1, c0);
    out[i] = z;
    ok[i]  = (thrust::abs(z - c0) < 4.0 * tk.half) ? 1 : 0;
    polyOut[i] = tk.poly;
}

static int nblocks(int n, int b) { return (n + b - 1) / b; }

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: batched_solve <N> <degree>\n"); return 1; }
    int N = std::atoi(argv[1]), deg = std::atoi(argv[2]), nc = deg + 1;
    if (N < 1 || deg < 1) { std::fprintf(stderr, "need N>=1, degree>=1\n"); return 1; }

    // read the batch: N polynomials, each nc "re im" pairs (descending), reversed
    // to ascending internally. (This I/O is NOT part of the timed solve.)
    std::vector<cmplx> h_coeffs((size_t)N * nc);
    for (int p = 0; p < N; ++p) {
        std::vector<cmplx> desc(nc);
        for (int k = 0; k < nc; ++k) { double re, im;
            if (!(std::cin >> re >> im)) { std::fprintf(stderr, "error: short input at poly %d\n", p); return 1; }
            desc[k] = cmplx(re, im); }
        for (int k = 0; k < nc; ++k) h_coeffs[(size_t)p*nc + k] = desc[nc-1-k];
    }

    const int    sps       = std::max(48, 4 * deg);
    const double isoThresh = std::min(0.5, 1.0 / deg), minHalf = 1e-6;
    const int    maxLevel  = 60;
    // Frontier peaks around N*deg (each poly ~deg active cells); 4x is safe
    // headroom. d_counts = maxTasks*64 ints dominates memory, so this bounds N.
    const size_t maxTasks  = (size_t)4 * N * deg + 4096;
    const size_t maxIso    = (size_t)N * deg + 1024;

    cmplx *d_coeffs, *d_deriv; double* d_bounds;
    BatchCell *d_front; int* d_counts;
    CK(cudaMalloc(&d_coeffs, (size_t)N * nc * sizeof(cmplx)));
    CK(cudaMalloc(&d_deriv,  (size_t)N * (nc-1) * sizeof(cmplx)));
    CK(cudaMalloc(&d_bounds, (size_t)N * sizeof(double)));
    CK(cudaMalloc(&d_front,  maxTasks * sizeof(BatchCell)));
    CK(cudaMalloc(&d_counts, maxTasks * NSUB * sizeof(int)));
    CK(cudaMemcpy(d_coeffs, h_coeffs.data(), (size_t)N * nc * sizeof(cmplx), cudaMemcpyHostToDevice));

    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a){ return std::chrono::duration<double,std::milli>(clk::now()-a).count(); };
    auto t0 = clk::now();
    double t_setup = 0, t_wind = 0, t_triage = 0, t_newton = 0;   // per-phase breakdown

    // --- parallel setup: derivatives, bounds, initial frontier (N tasks) ---
    auto s0 = clk::now();
    setup<<<nblocks(N, 256), 256>>>(d_coeffs, N, nc, d_deriv, d_bounds, d_front);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<BatchCell> frontier(N);
    CK(cudaMemcpy(frontier.data(), d_front, (size_t)N * sizeof(BatchCell), cudaMemcpyDeviceToHost));
    t_setup = ms_since(s0);

    // --- batched BFS over the shared (poly,cell) work-list ---
    std::vector<BatchCell> isolated;
    for (int lv = 0; lv < maxLevel && !frontier.empty(); ++lv) {
        size_t n = frontier.size();
        if (n > maxTasks) { std::fprintf(stderr, "task cap hit\n"); break; }
        auto g0 = clk::now();
        CK(cudaMemcpy(d_front, frontier.data(), n * sizeof(BatchCell), cudaMemcpyHostToDevice));
        batched_winding<<<(int)n, BLK>>>(d_front, d_coeffs, nc, sps, d_counts);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<int> counts(n * NSUB);
        CK(cudaMemcpy(counts.data(), d_counts, n * NSUB * sizeof(int), cudaMemcpyDeviceToHost));
        t_wind += ms_since(g0);                        // GPU winding + transfers

        auto h0 = clk::now();
        std::vector<BatchCell> next;
        for (size_t b = 0; b < n; ++b) {
            const BatchCell& tk = frontier[b]; double h = tk.half / K;
            for (int t = 0; t < NSUB; ++t) {
                int q = counts[b * NSUB + t]; if (q <= 0) continue;
                int i = t % K, j = t / K;
                BatchCell sub{ tk.poly, tk.cx - tk.half + (2*i+1)*h, tk.cy - tk.half + (2*j+1)*h, h };
                if      (q == 1 && sub.half <= isoThresh) isolated.push_back(sub);
                else if (sub.half <= minHalf)             { /* cluster: dropped in v1 */ }
                else                                      next.push_back(sub);
            }
        }
        frontier.swap(next);
        t_triage += ms_since(h0);                      // host-side triage (the suspect)
    }

    // --- batched Newton over all isolated cells ---
    auto nw0 = clk::now();
    std::vector<std::vector<cmplx>> roots(N);
    if (!isolated.empty() && isolated.size() <= maxIso) {
        int m = (int)isolated.size();
        cmplx* d_out; int *d_ok, *d_poly;
        CK(cudaMemcpy(d_front, isolated.data(), m * sizeof(BatchCell), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_out, m * sizeof(cmplx))); CK(cudaMalloc(&d_ok, m * sizeof(int)));
        CK(cudaMalloc(&d_poly, m * sizeof(int)));
        batched_newton<<<nblocks(m, BLK), BLK>>>((BatchCell*)d_front, m, d_coeffs, d_deriv, nc, d_out, d_ok, d_poly);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<cmplx> out(m); std::vector<int> ok(m), poly(m);
        CK(cudaMemcpy(out.data(),  d_out,  m * sizeof(cmplx), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ok.data(),   d_ok,   m * sizeof(int),   cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(poly.data(), d_poly, m * sizeof(int),   cudaMemcpyDeviceToHost));
        CK(cudaFree(d_out)); CK(cudaFree(d_ok)); CK(cudaFree(d_poly));
        for (int i = 0; i < m; ++i) {
            if (!ok[i]) continue; int p = poly[i]; bool dup = false;
            for (auto& r : roots[p]) if (thrust::abs(r - out[i]) < 1e-6) { dup = true; break; }
            if (!dup) roots[p].push_back(out[i]);
        }
    }

    t_newton = ms_since(nw0);
    double secs = ms_since(t0) / 1e3;

    // --- truth-free correctness: worst |P(root)| / max|coeff| over the batch ---
    double maxres = 0.0; size_t total = 0;
    for (int p = 0; p < N; ++p) {
        const cmplx* C = &h_coeffs[(size_t)p * nc];
        double scale = 0.0; for (int k = 0; k < nc; ++k) scale = fmax(scale, thrust::abs(C[k]));
        if (scale == 0) scale = 1;
        for (auto& r : roots[p]) { total++; maxres = fmax(maxres, thrust::abs(poly_eval(C, nc, r)) / scale); }
    }

    std::printf("SOLVE_MS %.3f\n", secs * 1e3);
    std::printf("SETUP_MS %.3f\n", t_setup);            // phase breakdown of SOLVE_MS
    std::printf("WINDING_MS %.3f\n", t_wind);           //   GPU winding kernels + transfers
    std::printf("TRIAGE_MS %.3f\n", t_triage);          //   host-side triage loop
    std::printf("NEWTON_MS %.3f\n", t_newton);
    std::printf("THROUGHPUT %.1f\n", N / secs);          // polynomials per second
    std::printf("MAXRESIDUAL %.2e\n", maxres);
    std::printf("ROOTS %zu   (expected %d)\n", total, N * deg);

    CK(cudaFree(d_coeffs)); CK(cudaFree(d_deriv)); CK(cudaFree(d_bounds));
    CK(cudaFree(d_front)); CK(cudaFree(d_counts));
    return 0;
}
