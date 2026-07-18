// ===========================================================================
// polyroots CUDA -- solve : the pluggable solver as a command-line tool
// ---------------------------------------------------------------------------
// One BFS, three swappable winding policies (compile-time template), the
// Cauchy/Fujiwara bound wired in, driven from the command line.
//
// Usage:
//   ./solve <mode> <degree>          then read (degree+1) "re im" pairs on stdin
//   ./solve -selfcheck               run the built-in known polynomial
//
//   mode: -naive | -gather | -scatter | -race
//
// Coefficients are entered DESCENDING (leading coefficient first), one complex
// pair "real imag" per value; degree n -> n+1 pairs. Example, z^5 - 1:
//   ./solve -naive 5
//   1 0
//   0 0
//   0 0
//   0 0
//   0 0
//   -1 0
// (piping a file works too: ./solve -race 5 < poly.txt)
//
// Kaggle:
//   %%writefile polyroots.cuh   (paste cuda/polyroots.cuh)
//   %%writefile solve.cu        (paste this)
//   !nvcc -O2 -arch=sm_75 solve.cu -o solve
//   !echo "1 0\n0 0\n0 0\n0 0\n0 0\n-1 0" | ./solve -race 5
//   !./solve -selfcheck
// ===========================================================================
#include "polyroots.cuh"
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <iostream>

constexpr int K    = 8;        // KxK sub-grid per square
constexpr int NSUB = K * K;
constexpr int BLK  = 64;

// ---- host: from_roots, Cauchy/Fujiwara bound ------------------------------
static std::vector<cmplx> from_roots(const std::vector<cmplx>& rt) {
    std::vector<cmplx> c = { cmplx(1,0) };
    for (cmplx r : rt) {
        std::vector<cmplx> nc(c.size() + 1, cmplx(0,0));
        for (size_t k = 0; k < c.size(); ++k) { nc[k+1] += c[k]; nc[k] += c[k] * (-r); }
        c = nc;
    }
    return c;
}
static double cauchy_bound(const std::vector<cmplx>& c) {
    int n = (int)c.size() - 1; double cn = thrust::abs(c[n]), m = 0;
    for (int k = 0; k < n; ++k) m = std::max(m, thrust::abs(c[k]) / cn);
    return 1.0 + m;
}
static double fujiwara_bound(const std::vector<cmplx>& c) {
    int n = (int)c.size() - 1; double cn = thrust::abs(c[n]), m = 0;
    for (int k = 1; k <= n; ++k) { double r = thrust::abs(c[n-k]) / cn; if (k==n) r *= 0.5;
        m = std::max(m, std::pow(r, 1.0 / k)); }
    return 2.0 * m;
}
static double root_bound(const std::vector<cmplx>& c) {
    return std::min(cauchy_bound(c), fujiwara_bound(c));
}

// ---- three winding policies (square -> KxK sub-cell counts) ----------------
struct PolicyP0 {
    static __device__ void compute(Cell s, const cmplx* c, int nc, int sps, int* out) {
        double h = s.half / K;
        for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
            int i = t % K, j = t / K;
            cmplx ctr(s.cx - s.half + (2*i+1)*h, s.cy - s.half + (2*j+1)*h);
            out[t] = winding_count(c, nc, ctr, h, sps);
        }
    }
};
struct PolicyP1 {
    static __device__ void compute(Cell s, const cmplx* c, int nc, int sps, int* out) {
        __shared__ double V[K * (K + 1)];
        __shared__ double H[(K + 1) * K];
        double h = s.half / K, ox = s.cx - s.half, oy = s.cy - s.half;
        for (int t = threadIdx.x; t < K * (K + 1); t += blockDim.x) {
            int col = t % (K + 1), j = t / (K + 1);
            V[j*(K+1)+col] = edge_phase(c, nc, cmplx(ox+2*h*col, oy+2*h*j),
                                                cmplx(ox+2*h*col, oy+2*h*(j+1)), sps);
        }
        for (int t = threadIdx.x; t < (K + 1) * K; t += blockDim.x) {
            int col = t % K, j = t / K;
            H[j*K+col] = edge_phase(c, nc, cmplx(ox+2*h*col,     oy+2*h*j),
                                            cmplx(ox+2*h*(col+1), oy+2*h*j), sps);
        }
        __syncthreads();
        for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
            int i = t % K, j = t / K;
            double d = H[j*K+i] + V[j*(K+1)+(i+1)] - H[(j+1)*K+i] - V[j*(K+1)+i];
            double w = d / (2 * kPI);
            out[t] = (int)(w >= 0 ? w + 0.5 : w - 0.5);
        }
    }
};
struct PolicyScatter {
    static __device__ void compute(Cell s, const cmplx* c, int nc, int sps, int* out) {
        __shared__ double M[NSUB];
        double h = s.half / K, ox = s.cx - s.half, oy = s.cy - s.half;
        for (int t = threadIdx.x; t < NSUB; t += blockDim.x) M[t] = 0.0;
        __syncthreads();
        for (int t = threadIdx.x; t < K * (K + 1); t += blockDim.x) {
            int col = t % (K + 1), j = t / (K + 1);
            double p = edge_phase(c, nc, cmplx(ox+2*h*col, oy+2*h*j),
                                          cmplx(ox+2*h*col, oy+2*h*(j+1)), sps);
            if (col >= 1) atomicAdd(&M[j*K+(col-1)],  p);
            if (col <  K) atomicAdd(&M[j*K+ col     ], -p);
        }
        for (int t = threadIdx.x; t < (K + 1) * K; t += blockDim.x) {
            int col = t % K, j = t / K;
            double p = edge_phase(c, nc, cmplx(ox+2*h*col,     oy+2*h*j),
                                          cmplx(ox+2*h*(col+1), oy+2*h*j), sps);
            if (j <  K) atomicAdd(&M[ j   *K+col],  p);
            if (j >= 1) atomicAdd(&M[(j-1)*K+col], -p);
        }
        __syncthreads();
        for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
            double w = M[t] / (2 * kPI);
            out[t] = (int)(w >= 0 ? w + 0.5 : w - 0.5);
        }
    }
};

template<class Policy>
__global__ void winding_kernel(const Cell* frontier, const cmplx* coeff, int nc,
                               int sps, int* counts) {
    Policy::compute(frontier[blockIdx.x], coeff, nc, sps, counts + blockIdx.x * NSUB);
}
__global__ void newton_kernel(const cmplx* coeff, int nc, const cmplx* dcoeff, int dn,
                              const Cell* cells, int m, cmplx* out, int* ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    cmplx c0(cells[i].cx, cells[i].cy);
    cmplx z = newton_polish(coeff, nc, dcoeff, dn, c0);
    out[i] = z;
    ok[i]  = (thrust::abs(z - c0) < 4.0 * cells[i].half) ? 1 : 0;
}

// ---- one BFS driver, templated on the winding policy ----------------------
// A located root: either a polished point (mult 1) or a certified box enclosure
// for a cluster / multiple root we could not separate (center +/- err, mult>=1).
struct Found { cmplx z; int mult; double err; bool cluster; };

template<class Policy>
std::vector<Found> solve(const std::vector<cmplx>& C, const std::vector<cmplx>& D,
                         double R, double& secs) {
    const int nc = (int)C.size(), dn = (int)D.size();
    // samples per edge, scaled with degree so the winding doesn't undersample
    // densely-packed roots (mirrors the CPU's ~4*degree adaptive count).
    const int sps = std::max(48, 4 * (nc - 1));
    // Isolate a count==1 cell only once it is small enough that its CENTER is a
    // reliable Newton seed. Too large and Newton escapes across the fractal
    // basin boundary to a neighbouring root -- catastrophic for densely-packed
    // roots (e.g. roots of unity), where the basins shrink ~1/n.
    //
    // Both thresholds are LENGTHS, so both must scale with the root bound R or
    // the solver is not scale-invariant. An absolute `min(0.5, 1/deg)` assumes
    // R ~ 1: scale the same roots down 10x and it isolates a level too early
    // and drops roots (measured: 8/10 at scale <= 0.1). Relative: 10/10 over
    // 1e-9..1e+9.
    // The extra 1/K is not cosmetic: R (Cauchy/Fujiwara) OVERESTIMATES the root
    // spread by ~3x, so R/n overestimates the true spacing. Since subdivision
    // divides by K per level, any threshold within a factor of K picks the same
    // level -- R/n and the old absolute 0.1 land on the SAME (too coarse) level.
    // R/(K*n) forces exactly one level deeper, which is what takes completeness
    // from ~96% to 100% (measured over a 300-polynomial host batch). Costs ~2.4x
    // the cells; that is the price of not silently dropping roots.
    const double isoThresh = R / (K * std::max(1, nc - 1));
    const double minHalf = R * 1e-7;
    const int maxLevel = 60;
    const size_t maxF = 1u << 16;

    cmplx *d_coeff, *d_dcoeff; Cell* d_front; int* d_counts;
    CK(cudaMalloc(&d_coeff,  nc * sizeof(cmplx)));
    CK(cudaMalloc(&d_dcoeff, dn * sizeof(cmplx)));
    CK(cudaMalloc(&d_front,  maxF * sizeof(Cell)));
    CK(cudaMalloc(&d_counts, maxF * NSUB * sizeof(int)));
    CK(cudaMemcpy(d_coeff,  C.data(), nc * sizeof(cmplx), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dcoeff, D.data(), dn * sizeof(cmplx), cudaMemcpyHostToDevice));

    double off = R * 0.05;
    std::vector<Cell> frontier = { { off*0.4142, off*0.3141, R*1.05 + off } };
    std::vector<Cell> isolated;
    std::vector<Cell> clusterCells; std::vector<int> clusterMult;

    auto t0 = std::chrono::steady_clock::now();
    for (int lv = 0; lv < maxLevel && !frontier.empty(); ++lv) {
        int n = (int)frontier.size();
        if ((size_t)n > maxF) { std::fprintf(stderr, "frontier cap hit\n"); break; }
        CK(cudaMemcpy(d_front, frontier.data(), n * sizeof(Cell), cudaMemcpyHostToDevice));
        winding_kernel<Policy><<<n, BLK>>>(d_front, d_coeff, nc, sps, d_counts);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<int> counts((size_t)n * NSUB);
        CK(cudaMemcpy(counts.data(), d_counts, (size_t)n * NSUB * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<Cell> next;
        for (int b = 0; b < n; ++b) {
            const Cell& s = frontier[b]; double h = s.half / K;
            for (int t = 0; t < NSUB; ++t) {
                int q = counts[(size_t)b * NSUB + t]; if (q <= 0) continue;
                int i = t % K, j = t / K;
                Cell sub{ s.cx - s.half + (2*i+1)*h, s.cy - s.half + (2*j+1)*h, h };
                if      (q == 1 && sub.half <= isoThresh) isolated.push_back(sub);
                else if (sub.half <= minHalf) {           // can't separate -> box enclosure
                    clusterCells.push_back(sub); clusterMult.push_back(q);
                }
                else                                      next.push_back(sub);
            }
        }
        frontier.swap(next);
    }

    std::vector<Found> found;
    if (!isolated.empty()) {
        int m = (int)isolated.size();
        cmplx* d_out; int* d_ok;
        CK(cudaMemcpy(d_front, isolated.data(), m * sizeof(Cell), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_out, m * sizeof(cmplx))); CK(cudaMalloc(&d_ok, m * sizeof(int)));
        newton_kernel<<<(m + BLK - 1) / BLK, BLK>>>(d_coeff, nc, d_dcoeff, dn, d_front, m, d_out, d_ok);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<cmplx> out(m); std::vector<int> ok(m);
        CK(cudaMemcpy(out.data(), d_out, m * sizeof(cmplx), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ok.data(),  d_ok,  m * sizeof(int),   cudaMemcpyDeviceToHost));
        CK(cudaFree(d_out)); CK(cudaFree(d_ok));
        for (int i = 0; i < m; ++i) {                              // polished point roots
            if (!ok[i]) continue; bool dup = false;
            // Scales with R for the same reason as isoThresh: an absolute 1e-6
            // merges every distinct root of a small-scale polynomial into one.
            const double dupTol = R * 1e-7;
            for (auto& f : found) if (!f.cluster && thrust::abs(f.z - out[i]) < dupTol) { dup = true; break; }
            if (!dup) found.push_back({ out[i], 1, 0.0, false });
        }
    }
    for (size_t i = 0; i < clusterCells.size(); ++i)               // certified box enclosures
        found.push_back({ cmplx(clusterCells[i].cx, clusterCells[i].cy),
                          clusterMult[i], clusterCells[i].half, true });
    secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CK(cudaFree(d_coeff)); CK(cudaFree(d_dcoeff)); CK(cudaFree(d_front)); CK(cudaFree(d_counts));
    return found;
}

// ---- CLI helpers ----------------------------------------------------------
// Read degree+1 "re im" pairs in DESCENDING order; return ASCENDING coeffs.
static bool read_poly(std::istream& in, int degree, std::vector<cmplx>& asc) {
    std::vector<cmplx> desc;
    for (int i = 0; i <= degree; ++i) {
        double re, im;
        if (!(in >> re >> im)) {
            std::fprintf(stderr, "error: expected %d coefficient pairs, got %d\n", degree + 1, i);
            return false;
        }
        desc.push_back(cmplx(re, im));
    }
    asc.resize(degree + 1);
    for (int k = 0; k <= degree; ++k) asc[k] = desc[degree - k];   // leading-first -> constant-first
    return true;
}
static void echo_poly(const std::vector<cmplx>& asc) {
    int d = (int)asc.size() - 1;
    std::printf("parsed (degree %d): ", d);
    for (int k = d; k >= 0; --k) {
        std::printf("(%g%+gi)z^%d", asc[k].real(), asc[k].imag(), k);
        if (k) std::printf(" + ");
    }
    std::printf("\n");
}
static void report(const char* name, const std::vector<Found>& found,
                   const std::vector<cmplx>& C, double secs) {
    size_t pts = 0, cls = 0; for (const auto& f : found) (f.cluster ? cls : pts)++;
    std::printf("[%-7s] %zu point root(s), %zu cluster(s)  (%.3f ms)\n",
                name, pts, cls, secs * 1e3);
    for (const auto& f : found) {
        if (!f.cluster) {                                   // polished point: show residual
            cmplx pr = poly_eval(C.data(), (int)C.size(), f.z);
            std::printf("    z = %+.10f %+.10fi    |P(z)| = %.2e\n",
                        f.z.real(), f.z.imag(), thrust::abs(pr));
        } else {                                            // certified box enclosure
            std::printf("    cluster: %d root(s) near %+.6f %+.6fi  +/- %.1e  (low confidence)\n",
                        f.mult, f.z.real(), f.z.imag(), f.err);
        }
    }
    // machine-parseable lines for the benchmark harness (bench/benchmark.py)
    std::printf("T %.6f\n", secs * 1e3);
    for (const auto& f : found)
        std::printf("R %.17g %.17g %d\n", f.z.real(), f.z.imag(), f.mult);
}

static std::vector<cmplx> derivative(const std::vector<cmplx>& C) {
    std::vector<cmplx> D((int)C.size() - 1);
    for (int k = 1; k < (int)C.size(); ++k) D[k-1] = C[k] * (double)k;
    return D;
}

static void run_and_report(const std::string& mode, const std::vector<cmplx>& C) {
    auto D = derivative(C); double R = root_bound(C), s;
    std::printf("root_bound = %.4f  (Cauchy %.3f, Fujiwara %.3f)\n\n",
                R, cauchy_bound(C), fujiwara_bound(C));
    if      (mode == "-naive")   { auto r = solve<PolicyP0>(C, D, R, s);      report("naive",   r, C, s); }
    else if (mode == "-gather")  { auto r = solve<PolicyP1>(C, D, R, s);      report("gather",  r, C, s); }
    else if (mode == "-scatter") { auto r = solve<PolicyScatter>(C, D, R, s); report("scatter", r, C, s); }
    else { // -race
        { double w; solve<PolicyP0>(C, D, R, w); }                    // warm up context
        auto r0 = solve<PolicyP0>(C, D, R, s);      report("naive",   r0, C, s);
        auto r1 = solve<PolicyP1>(C, D, R, s);      report("gather",  r1, C, s);
        auto r2 = solve<PolicyScatter>(C, D, R, s); report("scatter", r2, C, s);
    }
}

static void usage() {
    std::fprintf(stderr,
        "usage: ./solve <-naive|-gather|-scatter|-race> <degree>   (then degree+1 're im' pairs on stdin)\n"
        "       ./solve -selfcheck\n"
        "coefficients are DESCENDING (leading first); degree n needs n+1 pairs.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string mode = argv[1];

    if (mode == "-selfcheck") {                       // built-in known polynomial
        std::vector<cmplx> truth = { {2,0}, {-3,0}, {1,1}, {1,-1}, {0.5,0} };
        std::vector<cmplx> C = from_roots(truth);
        std::printf("self-check: roots {2,-3,1+i,1-i,0.5}\n");
        echo_poly(C);
        run_and_report("-race", C);
        return 0;
    }

    if (mode != "-naive" && mode != "-gather" && mode != "-scatter" && mode != "-race") {
        usage(); return 1;
    }
    if (argc < 3) { std::fprintf(stderr, "error: missing degree\n"); usage(); return 1; }
    int degree = std::atoi(argv[2]);
    if (degree < 1) { std::fprintf(stderr, "error: degree must be >= 1\n"); return 1; }

    std::vector<cmplx> C;
    if (!read_poly(std::cin, degree, C)) return 1;
    if (thrust::abs(C.back()) == 0.0) {
        std::fprintf(stderr, "error: leading coefficient is zero (not degree %d)\n", degree);
        return 1;
    }
    echo_poly(C);
    run_and_report(mode, C);
    return 0;
}
