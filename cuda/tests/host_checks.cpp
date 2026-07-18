// ===========================================================================
// Host-side regression checks for the CUDA solver's LEAF MATH.
// ---------------------------------------------------------------------------
// The CUDA kernels can only run on a GPU, but every numerical decision they
// make (winding, argument method, edge decomposition, thresholds) is ordinary
// double arithmetic. This file mirrors that math on the host so it can be
// compiled and run ANYWHERE -- no nvcc, no GPU -- and pins the properties that
// were expensive to discover:
//
//   1. SCALE INVARIANCE  -- every threshold compared against a length must
//      scale with the root bound R. Four absolute constants (isoThresh,
//      minHalf, the Newton step tolerance, the dedup tolerance) each broke
//      this independently; all four are exercised here over 1e-9 .. 1e+9.
//   2. ARG METHODS AGREE -- atan2 / approx-atan2 / quadrant-counting must find
//      the same roots (they are three ways to measure the same winding).
//   3. EDGE DECOMPOSITION -- the gather/scatter per-subcell counts must equal
//      naive's. This is where the edge ORIENTATION SIGNS get validated; a sign
//      error here is invisible until it silently miscounts on hardware.
//
// Build & run (exits non-zero on any failure):
//     g++ -O2 -std=c++17 cuda/tests/host_checks.cpp -o host_checks && ./host_checks
// ===========================================================================
#include <complex>
#include <vector>
#include <cstdio>
#include <cmath>
#include <functional>
#include <algorithm>

using cd = std::complex<double>;
constexpr double PI = 3.14159265358979323846;
constexpr int    K  = 8;              // KxK sub-grid per cell (matches the kernels)

// ---------------------------------------------------------------- leaf math --
static cd pe(const cd* c, int n, cd z) {                       // Horner
    cd b = c[n-1]; for (int k = n-2; k >= 0; --k) b = b*z + c[k]; return b;
}
static double lib_atan2(cd w) { return std::arg(w); }
static double approx_atan(double x) {                          // |x|<=1, ~1e-3 rad
    double ax = std::fabs(x);
    return PI/4*x - x*(ax - 1)*(0.2447 + 0.0663*ax);
}
static double approx_atan2(cd w) {
    double x = w.real(), y = w.imag(), ax = std::fabs(x), ay = std::fabs(y);
    if (ax == 0 && ay == 0) return 0;
    if (ax >= ay) { double a = approx_atan(y/x); if (x < 0) a += (y >= 0 ? PI : -PI); return a; }
    return (y >= 0 ? PI/2 : -PI/2) - approx_atan(x/y);
}
static int quad(cd w) {
    if (w.real() >= 0) return (w.imag() >= 0) ? 0 : 3;
    return (w.imag() >= 0) ? 1 : 2;
}

// Accumulated phase along A->B (sps steps). method 0=atan2 1=approx 2=quadrant.
// Angle methods return radians; quadrant returns net signed quarter-turns.
static double edge_phase(const cd* c, int nc, cd A, cd B, int sps, int method) {
    if (method == 2) {
        cd wp = pe(c, nc, A); int qp = quad(wp), net = 0;
        for (int k = 1; k <= sps; ++k) {
            cd w = pe(c, nc, A + (B-A)*(double(k)/sps));
            int q = quad(w), d = (q - qp + 4) % 4;
            if      (d == 1) net += 1;
            else if (d == 3) net -= 1;
            else if (d == 2) {                                 // half turn: sign from cross product
                double cr = wp.real()*w.imag() - wp.imag()*w.real();
                net += (cr >= 0) ? 2 : -2;
            }
            qp = q; wp = w;
        }
        return net;
    }
    auto ang = [&](cd w) { return method == 1 ? approx_atan2(w) : lib_atan2(w); };
    double prev = ang(pe(c, nc, A)), tot = 0;
    for (int k = 1; k <= sps; ++k) {
        double cur = ang(pe(c, nc, A + (B-A)*(double(k)/sps))), d = cur - prev;
        while (d <= -PI) d += 2*PI;
        while (d >   PI) d -= 2*PI;
        tot += d; prev = cur;
    }
    return tot;
}
static int finalize(double s, int method) {
    return (int)std::lround(method == 2 ? s/4.0 : s/(2*PI));
}
// Closed-loop winding of one square cell, by walking its 4 edges CCW.
static int wind_cell(const cd* c, int nc, double cx, double cy, double h, int sps, int method) {
    double X0 = cx-h, X1 = cx+h, Y0 = cy-h, Y1 = cy+h;
    double s = edge_phase(c, nc, cd(X0,Y0), cd(X1,Y0), sps, method)     // bottom  L->R
             + edge_phase(c, nc, cd(X1,Y0), cd(X1,Y1), sps, method)     // right   B->T
             + edge_phase(c, nc, cd(X1,Y1), cd(X0,Y1), sps, method)     // top     R->L
             + edge_phase(c, nc, cd(X0,Y1), cd(X0,Y0), sps, method);    // left    T->B
    return finalize(s, method);
}

// ------------------------------------------------------------ bounds/solve --
static double cauchy(const std::vector<cd>& c) {               // c ASCENDING
    int n = c.size()-1; double cn = std::abs(c[n]), m = 0;
    for (int k = 0; k < n; ++k) m = std::max(m, std::abs(c[k])/cn);
    return 1 + m;
}
static double fujiwara(const std::vector<cd>& c) {
    int n = c.size()-1; double cn = std::abs(c[n]), m = 0;
    for (int k = 1; k <= n; ++k) {
        double r = std::abs(c[n-k])/cn; if (k == n) r *= 0.5;
        m = std::max(m, std::pow(r, 1.0/k));
    }
    return 2*m;
}
static std::vector<cd> from_roots(const std::vector<cd>& rt) { // -> ASCENDING coeffs
    std::vector<cd> c = {{1,0}};
    for (cd r : rt) {
        std::vector<cd> n(c.size()+1, cd(0,0));
        for (size_t k = 0; k < c.size(); ++k) { n[k+1] += c[k]; n[k] += c[k]*(-r); }
        c = n;
    }
    return c;
}
static cd newton(const cd* c, int n, const cd* dc, int dn, cd z) {
    for (int i = 0; i < 80; ++i) {
        cd p = pe(c,n,z), d = pe(dc,dn,z);
        if (std::abs(d) < 1e-300) break;
        cd s = p/d; z -= s;
        if (std::abs(s) <= 1e-15*std::abs(z)) break;           // RELATIVE (see header)
    }
    return z;
}

struct Cell { double cx, cy, half; };

// Mirrors the CUDA BFS: subdivide KxK, isolate count==1 cells, Newton-polish.
// ALL thresholds are relative to the root bound R -- that is what check 1 pins.
static int solve_count(std::vector<cd> C, int method) {
    int nc = C.size(), deg = nc-1; const cd* c = C.data();
    std::vector<cd> D(deg);
    for (int k = 1; k <= deg; ++k) D[k-1] = C[k]*(double)k;

    double R    = std::min(cauchy(C), fujiwara(C));
    double off  = R*0.05;
    const int sps = std::max(48, 4*deg);
    const double isoT = R/deg;                                  // <-- relative
    const double minH = R*1e-7;                                 // <-- relative
    const double dupT = R*1e-7;                                 // <-- relative

    std::vector<Cell> frontier = { { off*0.4142, off*0.3141, R*1.05 + off } }, iso;
    for (int lv = 0; lv < 60 && !frontier.empty(); ++lv) {
        std::vector<Cell> next;
        for (Cell s : frontier) {
            double h = s.half/K;
            for (int j = 0; j < K; ++j) for (int i = 0; i < K; ++i) {
                double cx = s.cx - s.half + (2*i+1)*h, cy = s.cy - s.half + (2*j+1)*h;
                int q = wind_cell(c, nc, cx, cy, h, sps, method);
                if (q <= 0) continue;
                if      (q == 1 && h <= isoT) iso.push_back({cx,cy,h});
                else if (h <= minH)           { /* cluster: dropped, as in v1 */ }
                else                          next.push_back({cx,cy,h});
            }
        }
        frontier.swap(next);
    }
    std::vector<cd> roots;
    for (Cell s : iso) {
        cd z = newton(c, nc, D.data(), deg, cd(s.cx, s.cy));
        if (std::abs(z - cd(s.cx,s.cy)) < 4*s.half) {
            bool dup = false;
            for (auto& r : roots) if (std::abs(r - z) < dupT) { dup = true; break; }
            if (!dup) roots.push_back(z);
        }
    }
    return (int)roots.size();
}

// ------------------------------------------------------- gather / scatter ----
// Canonical edge tables for one parent cell: H(i,j) is L->R, V(i,j) is B->T.
static void build_tables(const cd* c, int nc, double cx, double cy, double H,
                         int sps, int method, std::vector<double>& Hd, std::vector<double>& Vd) {
    double h = H/K;
    auto X = [&](int i){ return cx - H + i*2*h; };
    auto Y = [&](int j){ return cy - H + j*2*h; };
    Hd.assign(K*(K+1), 0); Vd.assign((K+1)*K, 0);
    for (int j = 0; j <= K; ++j) for (int i = 0; i < K; ++i)
        Hd[j*K + i] = edge_phase(c, nc, cd(X(i),Y(j)), cd(X(i+1),Y(j)), sps, method);
    for (int j = 0; j < K; ++j) for (int i = 0; i <= K; ++i)
        Vd[j*(K+1) + i] = edge_phase(c, nc, cd(X(i),Y(j)), cd(X(i),Y(j+1)), sps, method);
}
// gather: each subcell PULLS its 4 oriented edges out of the tables.
static void gather_counts(const cd* c, int nc, double cx, double cy, double H,
                          int sps, int method, std::vector<int>& out) {
    std::vector<double> Hd, Vd; build_tables(c, nc, cx, cy, H, sps, method, Hd, Vd);
    out.assign(K*K, 0);
    for (int j = 0; j < K; ++j) for (int i = 0; i < K; ++i)
        out[j*K+i] = finalize(Hd[j*K+i] + Vd[j*(K+1)+(i+1)] - Hd[(j+1)*K+i] - Vd[j*(K+1)+i], method);
}
// scatter: each edge PUSHES its oriented phase into its (<=2) neighbour cells.
static void scatter_counts(const cd* c, int nc, double cx, double cy, double H,
                           int sps, int method, std::vector<int>& out) {
    std::vector<double> Hd, Vd; build_tables(c, nc, cx, cy, H, sps, method, Hd, Vd);
    std::vector<double> M(K*K, 0.0);
    for (int j = 0; j <= K; ++j) for (int i = 0; i < K; ++i) {
        double e = Hd[j*K + i];
        if (j < K) M[j*K + i]     += e;                        // bottom of (i,j)
        if (j > 0) M[(j-1)*K + i] -= e;                        // top    of (i,j-1)
    }
    for (int j = 0; j < K; ++j) for (int i = 0; i <= K; ++i) {
        double e = Vd[j*(K+1) + i];
        if (i < K) M[j*K + i]     -= e;                        // left   of (i,j)
        if (i > 0) M[j*K + (i-1)] += e;                        // right  of (i-1,j)
    }
    out.assign(K*K, 0);
    for (int t = 0; t < K*K; ++t) out[t] = finalize(M[t], method);
}

// ------------------------------------------------------------------ checks --
static int failures = 0;
static void report(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static std::vector<cd> base_roots() {
    return {{0.5,0.3},{-1,0.7},{1.2,-0.4},{-0.8,-1.1},{0.2,1.4},
            {-1.5,0.1},{0.9,0.9},{-0.3,-0.6},{1.1,0.2},{-0.6,1.0}};
}

// 1. Scale invariance: identical relative geometry at wildly different scales.
//    Regression for the four absolute-constant bugs. Absolute isoThresh gave
//    8/10 below scale 0.1; absolute dedup collapsed all 10 roots into 1 at 1e-9.
static void check_scale_invariance() {
    std::printf("1. scale invariance (same roots, scaled; expect 10/10 everywhere)\n");
    for (double s : {1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9}) {
        std::vector<cd> r; for (cd z : base_roots()) r.push_back(z*s);
        int n = solve_count(from_roots(r), 0);
        char buf[96]; std::snprintf(buf, sizeof buf, "scale %-8.0e -> %2d/10 roots", s, n);
        report(n == 10, buf);
    }
}
// 2. The three argument methods must agree (same winding, different measurement).
static void check_arg_methods() {
    std::printf("2. argument methods agree (atan2 / approx / quadrant)\n");
    std::vector<cd> unity(11, cd(0,0)); unity[0] = cd(-1,0); unity[10] = cd(1,0);
    struct { const char* name; std::vector<cd> C; } cases[] = {
        { "z^10 - 1",  unity },
        { "random d10", from_roots(base_roots()) },
    };
    for (auto& tc : cases)
        for (int m = 0; m < 3; ++m) {
            int n = solve_count(tc.C, m);
            char buf[96];
            std::snprintf(buf, sizeof buf, "%-11s method %d -> %2d/10 roots", tc.name, m, n);
            report(n == 10, buf);
        }
}
// 3. Edge decomposition: gather and scatter must reproduce naive EXACTLY.
//    This is where an edge orientation sign error would show up.
static void check_edge_decomposition() {
    std::printf("3. gather/scatter == naive over all 64 subcells\n");
    struct { const char* name; std::vector<cd> C; double cx, cy, H; } cases[] = {
        { "z^10 - 1",   {}, 0.02,  0.015, 2.1 },
        { "random d10", from_roots(base_roots()), 0.05, -0.03, 1.9 },
    };
    std::vector<cd> unity(11, cd(0,0)); unity[0] = cd(-1,0); unity[10] = cd(1,0);
    cases[0].C = unity;
    for (auto& tc : cases)
        for (int m = 0; m < 3; ++m) {
            int nc = tc.C.size(), sps = std::max(48, 4*(nc-1));
            std::vector<int> g, s; bool ok = true;
            gather_counts(tc.C.data(), nc, tc.cx, tc.cy, tc.H, sps, m, g);
            scatter_counts(tc.C.data(), nc, tc.cx, tc.cy, tc.H, sps, m, s);
            double h = tc.H/K;
            for (int j = 0; j < K && ok; ++j) for (int i = 0; i < K && ok; ++i) {
                double cx = tc.cx - tc.H + (2*i+1)*h, cy = tc.cy - tc.H + (2*j+1)*h;
                int n = wind_cell(tc.C.data(), nc, cx, cy, h, sps, m);
                if (n != g[j*K+i] || n != s[j*K+i]) ok = false;
            }
            char buf[96];
            std::snprintf(buf, sizeof buf, "%-11s method %d -> 64 subcells match", tc.name, m);
            report(ok, buf);
        }
}

int main() {
    std::printf("host regression checks for the CUDA leaf math\n\n");
    check_scale_invariance();  std::printf("\n");
    check_arg_methods();       std::printf("\n");
    check_edge_decomposition();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
