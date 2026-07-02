// ===========================================================================
// polyroots CUDA port -- Milestone P2: end-to-end subdivision solver
// ---------------------------------------------------------------------------
// Host-driven BFS over a work-list of cells (each loop = one kernel batch):
//   k_winding  -- per-cell winding count (one thread per frontier cell)
//   host triage -- drop empties, subdivide multi-root cells, collect isolated
//   k_polish   -- Newton-polish the isolated cells, flag Newton escapes
//
// The work-list lives on the HOST (small, irregular -> trivial to get right: no
// CUB, no device recursion, no atomics); the GPU does the expensive winding and
// Newton work. Each `while` iteration is one kernel launch over the frontier --
// the GPU form of the CPU solver's BFS. Shared math is in the header.
//
// Reproducibility note: GPU reduction/Newton order differs from the CPU, so
// compare roots with a tolerance, never bit-exact.
//
// Kaggle:
//   %%writefile polyroots.cuh          (paste cuda/polyroots.cuh)
//   %%writefile p2_subdivide_solve.cu  (paste this)
//   !nvcc -O2 -arch=sm_75 -lineinfo p2_subdivide_solve.cu -o p2 && ./p2
//   !compute-sanitizer ./p2
// ===========================================================================
#include "polyroots.cuh"
#include <vector>

// One thread per frontier cell: its winding-number root count.
__global__ void k_winding(const cmplx* coeff, int nc, const Cell* cells,
                          int ncells, int sps, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncells) return;
    counts[i] = winding_count(coeff, nc, cmplx(cells[i].cx, cells[i].cy),
                              cells[i].half, sps);
}

// One thread per isolated cell: Newton-polish from the centre, flag whether the
// result stayed near the cell (guards against Newton escaping to another root).
__global__ void k_polish(const cmplx* coeff, int nc, const cmplx* dcoeff, int dn,
                         const Cell* cells, int ncells, cmplx* out, int* ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncells) return;
    cmplx c0(cells[i].cx, cells[i].cy);
    cmplx z = newton_polish(coeff, nc, dcoeff, dn, c0);
    out[i] = z;
    ok[i]  = (thrust::abs(z - c0) < 4.0 * cells[i].half) ? 1 : 0;
}

static int launches(int n, int b) { return (n + b - 1) / b; }

int main() {
    // z^5 - 1 : five roots of unity.
    const int nc = 6;  cmplx h_coeff[nc]  = { {-1,0},{0,0},{0,0},{0,0},{0,0},{1,0} };
    const int dn = 5;  cmplx h_dcoeff[dn] = { {0,0},{0,0},{0,0},{0,0},{5,0} }; // 5 z^4
    const int degree = 5;

    const int    sps       = 32;
    const double isoThresh = 0.1;     // polish a count==1 cell once it is this small
    const double minHalf   = 1e-6;    // below this, an un-split count>=2 box is a cluster
    const int    maxLevel  = 40;
    const size_t maxCells  = 1u << 20;

    cmplx *d_coeff, *d_dcoeff;
    CK(cudaMalloc(&d_coeff,  nc * sizeof(cmplx)));
    CK(cudaMalloc(&d_dcoeff, dn * sizeof(cmplx)));
    CK(cudaMemcpy(d_coeff,  h_coeff,  nc * sizeof(cmplx), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dcoeff, h_dcoeff, dn * sizeof(cmplx), cudaMemcpyHostToDevice));

    Cell* d_cells;  int* d_counts;
    CK(cudaMalloc(&d_cells,  maxCells * sizeof(Cell)));
    CK(cudaMalloc(&d_counts, maxCells * sizeof(int)));

    double R = 1.0, half0 = R * 1.05 + 0.02, ox = 0.013, oy = 0.007;
    std::vector<Cell> frontier = { { ox, oy, half0 } };
    std::vector<Cell> isolated;
    int levels = 0, processed = 0, clusters = 0;

    // ---- host-driven BFS; each iteration is one k_winding launch ----
    for (int lv = 0; lv < maxLevel && !frontier.empty(); ++lv) {
        levels = lv + 1;
        int n = (int)frontier.size();
        if ((size_t)n > maxCells) { std::fprintf(stderr, "cap hit\n"); break; }

        CK(cudaMemcpy(d_cells, frontier.data(), n * sizeof(Cell), cudaMemcpyHostToDevice));
        k_winding<<<launches(n, 128), 128>>>(d_coeff, nc, d_cells, n, sps, d_counts);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

        std::vector<int> counts(n);
        CK(cudaMemcpy(counts.data(), d_counts, n * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<Cell> next;
        for (int i = 0; i < n; ++i) {
            ++processed;
            int cnt = counts[i];
            const Cell& cell = frontier[i];
            if (cnt <= 0) continue;                                   // empty -> drop
            if (cnt == 1 && cell.half <= isoThresh) { isolated.push_back(cell); continue; }
            if (cell.half <= minHalf) { ++clusters; continue; }       // unresolvable cluster
            double hh = cell.half * 0.5;                              // subdivide into 4
            next.push_back({ cell.cx - hh, cell.cy - hh, hh });
            next.push_back({ cell.cx + hh, cell.cy - hh, hh });
            next.push_back({ cell.cx + hh, cell.cy + hh, hh });
            next.push_back({ cell.cx - hh, cell.cy + hh, hh });
        }
        frontier.swap(next);
    }

    // ---- polish isolated cells on the GPU ----
    std::vector<cmplx> roots;
    if (!isolated.empty()) {
        int m = (int)isolated.size();
        cmplx* d_out; int* d_ok;
        CK(cudaMemcpy(d_cells, isolated.data(), m * sizeof(Cell), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_out, m * sizeof(cmplx)));
        CK(cudaMalloc(&d_ok,  m * sizeof(int)));
        k_polish<<<launches(m, 128), 128>>>(d_coeff, nc, d_dcoeff, dn, d_cells, m, d_out, d_ok);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

        std::vector<cmplx> out(m); std::vector<int> ok(m);
        CK(cudaMemcpy(out.data(), d_out, m * sizeof(cmplx), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ok.data(),  d_ok,  m * sizeof(int),   cudaMemcpyDeviceToHost));
        CK(cudaFree(d_out)); CK(cudaFree(d_ok));

        for (int i = 0; i < m; ++i) {                                // accept + dedup
            if (!ok[i]) continue;
            bool dup = false;
            for (auto& r : roots) if (thrust::abs(r - out[i]) < 1e-6) { dup = true; break; }
            if (!dup) roots.push_back(out[i]);
        }
    }

    // ---- report + validate against the known roots of unity ----
    std::printf("z^5 - 1 : levels=%d, processed=%d, isolated=%zu, clusters=%d\n",
                levels, processed, isolated.size(), clusters);
    std::printf("roots found: %zu (expected %d)\n", roots.size(), degree);
    double maxerr = 0.0;
    for (int k = 0; k < degree; ++k) {
        cmplx t = thrust::polar(1.0, 2 * kPI * k / degree);
        double best = 1e9;
        for (auto& r : roots) best = fmin(best, thrust::abs(r - t));
        maxerr = fmax(maxerr, best);
        std::printf("  truth %+.4f%+.4fi  nearest dist=%.2e\n", t.real(), t.imag(), best);
    }
    bool pass = (roots.size() == (size_t)degree) && (maxerr < 1e-8);
    std::printf("max error vs truth = %.2e  -> %s\n", maxerr, pass ? "PASS" : "FAIL");

    CK(cudaFree(d_coeff)); CK(cudaFree(d_dcoeff));
    CK(cudaFree(d_cells)); CK(cudaFree(d_counts));
    return 0;
}
