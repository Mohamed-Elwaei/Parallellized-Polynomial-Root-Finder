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
// (poly,cell) task; 64 threads cooperate on that cell's 8x8 sub-grid.
//
// I/O:  ./batched <N> <degree> [flags]   then N*(degree+1) "re im" pairs on stdin
//       (each polynomial's coeffs DESCENDING, polynomials back to back).
// Out:  PRECISION/ARG/POLICY (the config), SOLVE_MS, SETUP_MS/WINDING_MS/
//       TRIAGE_MS/NEWTON_MS (phase breakdown), THROUGHPUT, MAXRESIDUAL, ROOTS.
//
// Flags -- three orthogonal axes, all resolved at COMPILE time so the winding
// hot loop stays inlined (see launch_winding):
//   --float | --double            winding arithmetic precision  (default double)
//   --arg atan2|approx|quadrant   how the argument is computed   (default atan2)
//   --policy naive|gather|scatter how sub-grid edges are shared  (default naive)
//
// MEASURED (T4, N=10000, degree 10) -- ship --float --policy gather:
//   * --arg makes NO measurable difference: the winding is bound by the Horner
//     evaluation, not the transcendental. All three give identical roots.
//   * --policy DOES: gather 6.7x vs numpy, scatter 6.0x, naive 5.4x. gather beats
//     scatter because scatter's shared-memory atomics contend.
//   * --float is the big one: ~5x over --double (fp64 is 1/32 rate on a T4).
//
// Kaggle:
//   %%writefile polyroots.cuh     (paste cuda/polyroots.cuh)
//   %%writefile batched_solve.cu  (paste this)
//   !nvcc -O2 -std=c++17 -arch=sm_75 batched_solve.cu -o batched   (C++17 needed
//                                    for the `if constexpr` dispatch)
// ===========================================================================
#include "polyroots.cuh"
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>

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

// ---- winding templated on the compute precision T (float or double) ---------
// Coefficients are stored in double; each is cast to T on read, so the ARITHMETIC
// (Horner + arg/atan2) runs in T. On fp64-slow GPUs (e.g. T4) T=float is much
// faster, and it's precise enough to COUNT roots (Newton still runs in double).
template<class T>
__device__ inline thrust::complex<T> heval_T(const cmplx* coeff, int nc, thrust::complex<T> z) {
    thrust::complex<T> b(T(coeff[nc-1].real()), T(coeff[nc-1].imag()));
    for (int k = nc - 2; k >= 0; --k)
        b = b * z + thrust::complex<T>(T(coeff[k].real()), T(coeff[k].imag()));
    return b;
}
template<class T>
__device__ inline thrust::complex<T> perim_T(T cx, T cy, T half, int sps, int i) {
    int s = i / sps; T t = T(i % sps) / sps, h = half, x, y;
    if      (s == 0) { x = -h + 2*h*t; y = -h; }
    else if (s == 1) { x =  h;         y = -h + 2*h*t; }
    else if (s == 2) { x =  h - 2*h*t; y =  h; }
    else             { x = -h;         y =  h - 2*h*t; }
    return thrust::complex<T>(cx + x, cy + y);
}
// ---- argument methods (selectable via --arg) --------------------------------
// (0) library atan2  : full-precision reference (thrust::arg).
// (1) approx atan2   : cheap polynomial atan2, ~1e-3 rad; loop-telescoping
//                      cancels the error so the winding is still exact.
// (2) quadrant count : sign-based; no transcendental at all.
template<class T> struct LibAtan2 {
    __device__ T operator()(thrust::complex<T> w) const { return thrust::arg(w); }
};
// atan(r) for |r|<=1, standard cheap fit: max error ~1e-3 rad for ~4 multiplies
// (vs the library's full-precision argument reduction + high-degree polynomial).
//
// Why such a sloppy approximation is safe here: the winding sums DIFFERENCES of
// angles around a CLOSED loop. Writing the approximation error as e(theta) (it
// depends only on the angle, since atan2 is scale-invariant), each step
// contributes (e_{k+1} - e_k), and summing around the loop TELESCOPES to
// e_last - e_first = 0 -- the loop returns to its starting point, so the errors
// cancel exactly. The approximation therefore does not perturb the winding at
// all; it only has to be accurate enough never to flip an unwrap decision (push
// a step past +-pi), and ~1e-3 rad is orders of magnitude inside that margin.
template<class T> __device__ inline T approx_atan_unit(T r) {   // atan(r), |r|<=1
    const T PI = T(3.14159265358979323846);
    T ar = r < 0 ? -r : r;
    return T(0.25) * PI * r - r * (ar - T(1)) * (T(0.2447) + T(0.0663) * ar);
}
template<class T> struct ApproxAtan2 {
    __device__ T operator()(thrust::complex<T> w) const {
        const T PI = T(3.14159265358979323846), HP = PI * T(0.5);
        T x = w.real(), y = w.imag();
        T ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
        if (ax == 0 && ay == 0) return T(0);
        if (ax >= ay) { T a = approx_atan_unit<T>(y / x); if (x < 0) a += (y >= 0 ? PI : -PI); return a; }
        return (y >= 0 ? HP : -HP) - approx_atan_unit<T>(x / y);
    }
};

// Winding via ANGLE accumulation — one impl serving atan2 AND approx (swap AngleFn).
template<class T, class AngleFn>
__device__ int wind_angle(const cmplx* coeff, int nc, T cx, T cy, T half, int sps) {
    AngleFn ang;
    const T PI = T(3.14159265358979323846);
    int S = sps * 4;
    T prev = ang(heval_T<T>(coeff, nc, perim_T<T>(cx, cy, half, sps, 0)));
    T tot = 0;
    for (int i = 1; i <= S; ++i) {
        T cur = ang(heval_T<T>(coeff, nc, perim_T<T>(cx, cy, half, sps, i % S)));
        T d = cur - prev;
        while (d <= -PI) d += 2 * PI;
        while (d >   PI) d -= 2 * PI;
        tot += d; prev = cur;
    }
    double w = (double)tot / (2 * 3.14159265358979323846);
    return (int)(w >= 0 ? w + 0.5 : w - 0.5);
}
// Winding via QUADRANT counting -- no transcendental at all.
// Instead of measuring the angle, just track which quadrant P(z) is in. Each
// quadrant crossing is a quarter turn (pi/2), so a full revolution = 4 signed
// crossings, and winding = (net crossings)/4 -- an exact integer, no rounding
// of a float sum. Per step, d = (q_new - q_old) mod 4 gives:
//   d=0 -> stayed put   d=1 -> +1 quarter (CCW)   d=3 -> -1 quarter (CW)
//   d=2 -> AMBIGUOUS: a half turn is +2 or -2 quarters and the quadrant index
//          alone cannot distinguish them. Resolve with the 2D cross product
//          cr = re(a)*im(b) - im(a)*re(b) = |a||b|sin(phi): its SIGN is the sign
//          of the rotation angle, so cr >= 0 means CCW (+2), else CW (-2).
// (Sampling is dense enough that |d|>2 -- more than a half turn in one step --
// does not occur; that would be undersampling, the same failure mode the angle
// methods have at the +-pi unwrap boundary.)
template<class T> __device__ inline int quadrant_of(thrust::complex<T> w) {
    if (w.real() >= 0) return (w.imag() >= 0) ? 0 : 3;
    return (w.imag() >= 0) ? 1 : 2;
}
template<class T>
__device__ int wind_quadrant(const cmplx* coeff, int nc, T cx, T cy, T half, int sps) {
    int S = sps * 4;
    thrust::complex<T> wprev = heval_T<T>(coeff, nc, perim_T<T>(cx, cy, half, sps, 0));
    int qprev = quadrant_of<T>(wprev), net = 0;
    for (int i = 1; i <= S; ++i) {
        thrust::complex<T> w = heval_T<T>(coeff, nc, perim_T<T>(cx, cy, half, sps, i % S));
        int q = quadrant_of<T>(w), d = (q - qprev + 4) % 4;
        if      (d == 1) net += 1;
        else if (d == 3) net -= 1;
        else if (d == 2) { T cr = wprev.real()*w.imag() - wprev.imag()*w.real(); net += (cr >= 0) ? 2 : -2; }
        qprev = q; wprev = w;
    }
    double q = net / 4.0;
    return (int)(q >= 0 ? q + 0.5 : q - 0.5);
}
// Compile-time method pick (0 atan2, 1 approx, 2 quadrant) — inlined, no runtime branch.
template<class T, int METHOD>
__device__ inline int winding_dispatch(const cmplx* C, int nc, T cx, T cy, T h, int sps) {
    if      constexpr (METHOD == 2) return wind_quadrant<T>(C, nc, cx, cy, h, sps);
    else if constexpr (METHOD == 1) return wind_angle<T, ApproxAtan2<T>>(C, nc, cx, cy, h, sps);
    else                            return wind_angle<T, LibAtan2<T>>(C, nc, cx, cy, h, sps);
}

// ===== winding POLICIES (selectable via --policy) ============================
// naive : one thread per subcell, samples its OWN 4 edges (interior edges are
//         recomputed by both neighbours).
// gather: threads fill a shared canonical edge table ONCE, then each subcell
//         READS its 4 edges from it (no atomics; each shared edge evaluated once).
// scatter: threads compute each edge once and atomicAdd its oriented contribution
//         into the two adjacent subcells' shared accumulator (nondeterministic add).
// gather/scatter do ~1.7x fewer Horner evals than naive (144 edges vs 256).

// --- NAIVE ---
template<class T, int METHOD>
__global__ void batched_winding_naive(const BatchCell* frontier, const cmplx* coeffs,
                                      int nc, int sps, int* counts) {
    int b = blockIdx.x;
    BatchCell tk = frontier[b];
    const cmplx* C = coeffs + (size_t)tk.poly * nc;           // <-- this poly's row
    T h = T(tk.half) / K;
    for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
        int i = t % K, j = t / K;
        T ctrx = T(tk.cx) - T(tk.half) + (2*i+1) * h;
        T ctry = T(tk.cy) - T(tk.half) + (2*j+1) * h;
        counts[(size_t)b * NSUB + t] = winding_dispatch<T, METHOD>(C, nc, ctrx, ctry, h, sps);
    }
}

// Accumulated phase along segment A->B (sps steps). Angle methods return radians;
// quadrant returns net signed transitions. Both are path-additive and negate on
// reversal, so a subcell winding = sum of its 4 oriented edges.
template<class T, int METHOD>
__device__ T edge_phase(const cmplx* C, int nc, T ax, T ay, T bx, T by, int sps) {
    if constexpr (METHOD == 2) {
        thrust::complex<T> wp = heval_T<T>(C, nc, thrust::complex<T>(ax, ay));
        int qp = quadrant_of<T>(wp), net = 0;
        for (int k = 1; k <= sps; ++k) {
            T t = T(k) / sps;
            thrust::complex<T> w = heval_T<T>(C, nc, thrust::complex<T>(ax + (bx-ax)*t, ay + (by-ay)*t));
            int q = quadrant_of<T>(w), d = (q - qp + 4) % 4;
            if      (d == 1) net += 1;
            else if (d == 3) net -= 1;
            else if (d == 2) { T cr = wp.real()*w.imag() - wp.imag()*w.real(); net += (cr >= 0) ? 2 : -2; }
            qp = q; wp = w;
        }
        return T(net);
    } else {
        using AngleFn = typename std::conditional<METHOD == 1, ApproxAtan2<T>, LibAtan2<T>>::type;
        AngleFn ang; const T PI = T(3.14159265358979323846);
        T prev = ang(heval_T<T>(C, nc, thrust::complex<T>(ax, ay))), tot = 0;
        for (int k = 1; k <= sps; ++k) {
            T t = T(k) / sps;
            T cur = ang(heval_T<T>(C, nc, thrust::complex<T>(ax + (bx-ax)*t, ay + (by-ay)*t)));
            T d = cur - prev;
            while (d <= -PI) d += 2 * PI;
            while (d >   PI) d -= 2 * PI;
            tot += d; prev = cur;
        }
        return tot;
    }
}
// Turn an accumulated subcell sum into an integer root count. The divisor is
// "one full revolution" in that method's units: 2*pi radians for the angle
// methods, 4 quarter-turn crossings for quadrant counting.
template<class T, int METHOD>
__device__ inline int edge_finalize(T s) {
    double v = (METHOD == 2) ? (double)s / 4.0 : (double)s / (2 * 3.14159265358979323846);
    return (int)(v >= 0 ? v + 0.5 : v - 0.5);
}

// grid line coordinate helpers: the KxK sub-grid is cut by K+1 lines per axis,
// so corner (i,j) = (gridX(i), gridY(j)) for i,j in [0,K].
template<class T> __device__ inline T gridX(const BatchCell& tk, T h, int i) { return T(tk.cx) - T(tk.half) + i * 2 * h; }
template<class T> __device__ inline T gridY(const BatchCell& tk, T h, int j) { return T(tk.cy) - T(tk.half) + j * 2 * h; }

// --- the edge-sharing identity (why gather/scatter work) --------------------
// Every edge is stored ONCE in a canonical direction: horizontal L->R, vertical
// B->T. Subcell (i,j) spans corners (i,j)..(i+1,j+1); walking its boundary
// COUNTER-CLOCKWISE visits:
//
//        (i,j+1)  <--- H(i,j+1) ---  (i+1,j+1)      bottom = H(i,j)      (+, L->R = canonical)
//           |                             ^         right  = V(i+1,j)    (+, B->T = canonical)
//        V(i,j)                      V(i+1,j)       top    = H(i,j+1)    (-, walked R->L)
//           v                             |         left   = V(i,j)      (-, walked T->B)
//        (i,j)    ---  H(i,j)   --->  (i+1,j)
//
// Reversing a path negates its accumulated phase, hence the two minus signs:
//
//     winding(i,j) = H(i,j) + V(i+1,j) - H(i,j+1) - V(i,j)
//
// Each interior edge is shared by two neighbours with OPPOSITE signs, so it
// cancels exactly -- which is both why one evaluation suffices (1.7x fewer
// Horner evals: 144 edges vs naive's 256) and why the shared boundary is more
// numerically consistent than naive's two independent recomputations.

// --- GATHER: shared canonical edge tables, each subcell reads its 4 edges ---
template<class T, int METHOD>
__global__ void batched_winding_gather(const BatchCell* frontier, const cmplx* coeffs,
                                       int nc, int sps, int* counts) {
    int b = blockIdx.x;
    BatchCell tk = frontier[b];
    const cmplx* C = coeffs + (size_t)tk.poly * nc;
    T h = T(tk.half) / K;
    __shared__ T Hd[K * (K + 1)];    // horizontal edges (L->R), idx = j*K + i, i in [0,K), j in [0,K]
    __shared__ T Vd[(K + 1) * K];    // vertical   edges (B->T), idx = j*(K+1) + i, i in [0,K], j in [0,K)
    for (int e = threadIdx.x; e < K * (K + 1); e += blockDim.x) {
        int i = e % K, j = e / K;
        Hd[e] = edge_phase<T, METHOD>(C, nc, gridX<T>(tk,h,i), gridY<T>(tk,h,j), gridX<T>(tk,h,i+1), gridY<T>(tk,h,j), sps);
    }
    for (int e = threadIdx.x; e < (K + 1) * K; e += blockDim.x) {
        int i = e % (K + 1), j = e / (K + 1);
        Vd[e] = edge_phase<T, METHOD>(C, nc, gridX<T>(tk,h,i), gridY<T>(tk,h,j), gridX<T>(tk,h,i), gridY<T>(tk,h,j+1), sps);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < NSUB; t += blockDim.x) {
        int i = t % K, j = t / K;
        T s = Hd[j*K + i] + Vd[j*(K+1) + (i+1)] - Hd[(j+1)*K + i] - Vd[j*(K+1) + i];
        counts[(size_t)b * NSUB + t] = edge_finalize<T, METHOD>(s);
    }
}

// --- SCATTER: each edge pushes its oriented phase into its <=2 neighbour cells ---
template<class T, int METHOD>
__global__ void batched_winding_scatter(const BatchCell* frontier, const cmplx* coeffs,
                                        int nc, int sps, int* counts) {
    int b = blockIdx.x;
    BatchCell tk = frontier[b];
    const cmplx* C = coeffs + (size_t)tk.poly * nc;
    T h = T(tk.half) / K;
    __shared__ T M[NSUB];
    for (int t = threadIdx.x; t < NSUB; t += blockDim.x) M[t] = T(0);
    __syncthreads();
    for (int e = threadIdx.x; e < K * (K + 1); e += blockDim.x) {   // horizontal edges
        int i = e % K, j = e / K;
        T ph = edge_phase<T, METHOD>(C, nc, gridX<T>(tk,h,i), gridY<T>(tk,h,j), gridX<T>(tk,h,i+1), gridY<T>(tk,h,j), sps);
        if (j < K) atomicAdd(&M[j*K + i], ph);          // bottom of (i,j)
        if (j > 0) atomicAdd(&M[(j-1)*K + i], -ph);     // top of (i,j-1)
    }
    for (int e = threadIdx.x; e < (K + 1) * K; e += blockDim.x) {   // vertical edges
        int i = e % (K + 1), j = e / (K + 1);
        T ph = edge_phase<T, METHOD>(C, nc, gridX<T>(tk,h,i), gridY<T>(tk,h,j), gridX<T>(tk,h,i), gridY<T>(tk,h,j+1), sps);
        if (i < K) atomicAdd(&M[j*K + i], -ph);         // left of (i,j)
        if (i > 0) atomicAdd(&M[j*K + (i-1)], ph);      // right of (i-1,j)
    }
    __syncthreads();
    for (int t = threadIdx.x; t < NSUB; t += blockDim.x)
        counts[(size_t)b * NSUB + t] = edge_finalize<T, METHOD>(M[t]);
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

// Launch the winding for a fixed (precision T, arg METHOD), picking the policy
// kernel at run time (host-side branch, once per BFS level — negligible). Each
// kernel is a distinct specialization, so the device hot loop stays inlined.
template<class T, int METHOD>
static void launch_winding(int policy, size_t n, const BatchCell* d_front,
                           const cmplx* d_coeffs, int nc, int sps, int* d_counts) {
    if      (policy == 1) batched_winding_gather <T,METHOD><<<(int)n, BLK>>>(d_front, d_coeffs, nc, sps, d_counts);
    else if (policy == 2) batched_winding_scatter<T,METHOD><<<(int)n, BLK>>>(d_front, d_coeffs, nc, sps, d_counts);
    else                  batched_winding_naive  <T,METHOD><<<(int)n, BLK>>>(d_front, d_coeffs, nc, sps, d_counts);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr,
        "usage: batched_solve <N> <degree> [--float|--double] [--arg atan2|approx|quadrant] "
        "[--policy naive|gather|scatter]\n"); return 1; }
    int N = std::atoi(argv[1]), deg = std::atoi(argv[2]), nc = deg + 1;
    if (N < 1 || deg < 1) { std::fprintf(stderr, "need N>=1, degree>=1\n"); return 1; }
    bool useFloat = false;                                    // winding precision (--float|--double)
    int  argMethod = 0;                                       // 0 atan2, 1 approx, 2 quadrant (--arg)
    int  policy    = 0;                                       // 0 naive, 1 gather, 2 scatter (--policy)
    for (int a = 3; a < argc; ++a) {
        std::string s = argv[a];
        if      (s == "--float")  useFloat = true;
        else if (s == "--double") useFloat = false;
        else if (s == "--arg" && a + 1 < argc) {
            std::string m = argv[++a];
            if      (m == "atan2")    argMethod = 0;
            else if (m == "approx")   argMethod = 1;
            else if (m == "quadrant") argMethod = 2;
            else { std::fprintf(stderr, "unknown --arg '%s' (atan2|approx|quadrant)\n", m.c_str()); return 1; }
        }
        else if (s == "--policy" && a + 1 < argc) {
            std::string m = argv[++a];
            if      (m == "naive")   policy = 0;
            else if (m == "gather")  policy = 1;
            else if (m == "scatter") policy = 2;
            else { std::fprintf(stderr, "unknown --policy '%s' (naive|gather|scatter)\n", m.c_str()); return 1; }
        }
    }

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

    // Samples per edge, scaled with degree so the winding doesn't undersample
    // densely-packed roots (mirrors the CPU's ~4*degree adaptive count).
    const int    sps       = std::max(48, 4 * deg);
    // isoThresh / minHalf are computed PER POLYNOMIAL after setup (they scale
    // with that polynomial's root bound R) -- see the isoT/minH arrays below.
    //
    // isoThresh: isolate a count==1 cell only once it is small enough that its
    // CENTER is a reliable Newton seed. From a too-large cell Newton escapes
    // across the fractal basin boundary to a NEIGHBOURING root, the inside-check
    // then rejects it, and the root is lost entirely. Roots inside a bound R
    // are spaced ~R/n, and the seed error is the cell half-diagonal, so the
    // criterion must be half << R/n -- hence isoThresh = R/deg.
    //
    // The R matters: an absolute `min(0.5, 1.0/deg)` has the right 1/n scaling
    // but DROPS the length unit, silently assuming R ~ 1. It compares a length
    // to a dimensionless number. Roots of unity (R ~ 2) happen to sit where
    // that is accidentally correct; scale the same roots down by 10x and the
    // solver isolates a level too early and loses roots. Measured on host:
    // absolute -> 8/10 roots at scale <= 0.1; relative -> 10/10 over 1e-9..1e+9.
    const int    maxLevel  = 60;
    // Frontier peaks around N*deg (each poly ~deg active cells); 4x is safe
    // headroom. d_counts = maxTasks*64 ints dominates memory, so this bounds N.
    const size_t maxTasks  = (size_t)4 * N * deg + 4096;

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
    // Per-polynomial thresholds. Every threshold that is COMPARED AGAINST A
    // LENGTH must itself scale with that polynomial's root bound R, or the
    // solver stops being scale-invariant (see the isoThresh note above). R
    // differs per polynomial here, so these are arrays, not scalars.
    std::vector<double> h_bounds(N);
    CK(cudaMemcpy(h_bounds.data(), d_bounds, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> isoT(N), minH(N);
    // The extra 1/K: R overestimates the root spread by ~3x, and subdivision
    // divides by K per level, so R/deg and the old absolute 0.1 select the SAME
    // (too coarse) level -- both isolate cells that are still too big to seed
    // Newton reliably. R/(K*deg) forces one level deeper: completeness 96% ->
    // 100% on a 300-polynomial host batch, at ~2.4x the cells.
    for (int p = 0; p < N; ++p) { isoT[p] = h_bounds[p] / (K * deg); minH[p] = h_bounds[p] * 1e-7; }

    // --- --float representability guard ------------------------------------
    // Coefficients are STORED in double but CAST TO FLOAT for the winding when
    // --float is selected. The test is simply whether each coefficient survives
    // that cast: |c_0| ~ (root magnitude)^deg, so the usable root magnitude is a
    // BAND that narrows as degree rises (deg 20 -> [0.013, 84], deg 100 ->
    // [0.42, 2.4]). Outside it a coefficient becomes inf or flushes to zero and
    // the winding is garbage, with no other signal anywhere in the pipeline.
    //
    // NOTE: the test is on the COEFFICIENTS, not on |P(z)| over the contour.
    // Measured: intermediate overflow far from the roots is benign (those
    // regions have winding 0, so an error there costs work, not roots), and a
    // contour-based bound raised false alarms on batches that solved perfectly.
    // Underflow is the dangerous side -- it zeroes P near the roots too, and a
    // scale-1e-6 batch silently returned 0.04% of its roots.
    if (useFloat) {
        const double kFMax = 3.4028235e38, kFMin = 1.1754944e-38;
        int over = 0, under = 0; double mx = 0, mn = 1e308;
        for (int p = 0; p < N; ++p) {
            const cmplx* C = &h_coeffs[(size_t)p * nc];
            bool o = false, u = false;
            for (int k = 0; k < nc; ++k) {
                double a = thrust::abs(C[k]);
                if (a > kFMax)             o = true;
                if (a > 0 && a < kFMin)    u = true;
                if (a > mx)                mx = a;
                if (a > 0 && a < mn)       mn = a;
            }
            if (o) ++over;
            if (u) ++under;
        }
        if (over)
            std::fprintf(stderr, "warning: --float OVERFLOW on %d/%d polynomial(s): "
                "largest coefficient %.3g exceeds float max %.3g -- roots are too "
                "large for this degree. Rerun with --double.\n", over, N, mx, kFMax);
        if (under)
            std::fprintf(stderr, "warning: --float UNDERFLOW on %d/%d polynomial(s): "
                "smallest non-zero coefficient %.3g is below float min %.3g and will "
                "flush to zero -- roots are too small for this degree. Rerun with "
                "--double.\n", under, N, mn, kFMin);
    }
    t_setup = ms_since(s0);

    // --- batched BFS over the shared (poly,cell) work-list ---
    std::vector<BatchCell> isolated;
    for (int lv = 0; lv < maxLevel && !frontier.empty(); ++lv) {
        size_t n = frontier.size();
        if (n > maxTasks) { std::fprintf(stderr, "task cap hit\n"); break; }
        auto g0 = clk::now();
        CK(cudaMemcpy(d_front, frontier.data(), n * sizeof(BatchCell), cudaMemcpyHostToDevice));
        // dispatch {float,double} x {atan2,approx,quadrant} at compile time;
        // launch_winding then picks the policy kernel (naive|gather|scatter).
        #define LAUNCH(T,M) launch_winding<T,M>(policy, n, d_front, d_coeffs, nc, sps, d_counts)
        if (useFloat) { if (argMethod==2) LAUNCH(float,2);  else if (argMethod==1) LAUNCH(float,1);  else LAUNCH(float,0); }
        else          { if (argMethod==2) LAUNCH(double,2); else if (argMethod==1) LAUNCH(double,1); else LAUNCH(double,0); }
        #undef LAUNCH
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
                if      (q == 1 && sub.half <= isoT[tk.poly]) isolated.push_back(sub);
                else if (sub.half <= minH[tk.poly])          { /* cluster: dropped in v1 */ }
                else                                         next.push_back(sub);
            }
        }
        frontier.swap(next);
        t_triage += ms_since(h0);                      // host-side triage (the suspect)
    }

    // --- batched Newton over all isolated cells ---
    auto nw0 = clk::now();
    std::vector<std::vector<cmplx>> roots(N);
    if (isolated.size() > maxTasks) {                        // reuse d_front; guard its size
        std::fprintf(stderr, "warning: %zu isolated > buffer %zu; truncating\n",
                     isolated.size(), maxTasks);
        isolated.resize(maxTasks);
    }
    if (!isolated.empty()) {
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
            // Dedup tolerance must ALSO scale with R: an absolute 1e-6 merges
            // every distinct root of a small-scale polynomial into one (at
            // scale 1e-9 all ten roots are within 1e-6 of each other -> 1 root).
            const double dupTol = h_bounds[p] * 1e-7;
            for (auto& r : roots[p]) if (thrust::abs(r - out[i]) < dupTol) { dup = true; break; }
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

    std::printf("PRECISION %s\n", useFloat ? "float" : "double");
    std::printf("ARG %s\n", argMethod == 2 ? "quadrant" : argMethod == 1 ? "approx" : "atan2");
    std::printf("POLICY %s\n", policy == 2 ? "scatter" : policy == 1 ? "gather" : "naive");
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
