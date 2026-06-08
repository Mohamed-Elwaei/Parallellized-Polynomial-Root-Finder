// poly_roots_cpu.cpp
// ---------------------------------------------------------------------------
// Phase 1: single-threaded CPU reference for a complex-analytic
// (argument-principle / winding-number) subdivision polynomial root-finder.
//
// This is the correctness oracle and the CPU baseline that the CUDA port
// (Phase 3) will be validated and benchmarked against. The control flow is
// deliberately written as a breadth-first work-list (process the whole active
// set, then swap in the children) so it maps directly onto the GPU design:
// "each kernel launch processes the entire current queue in parallel."
//
// Build:  g++ -O2 -std=c++17 poly_roots_cpu.cpp -o poly_roots_cpu
// Run:    ./poly_roots_cpu
// ---------------------------------------------------------------------------

#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <string>
#include <chrono>

using cd = std::complex<double>;
static const double PI     = 3.14159265358979323846;
static const double TWO_PI = 2.0 * PI;

// ===========================================================================
// Polynomial
// ===========================================================================
struct Polynomial {
    // coeff[k] is the coefficient of z^k. coeff.back() is the leading term c_n.
    std::vector<cd> coeff;

    int degree() const { return (int)coeff.size() - 1; }

    // Horner evaluation. Plain double precision is fine for moderate degree.
    // For very high degree / very large |z| the intermediate |b_k| can overflow
    // a double (~1.8e308); that is exactly what the "rescaling Horner" variant
    // in the roadmap fixes, and it becomes relevant in the CUDA / high-degree
    // phase. For the Phase-1 validation polynomials it is not needed, *provided*
    // we use a tight root bound (see below) so the contour never reaches a
    // radius where P overflows.
    cd eval(cd z) const {
        cd b = coeff.back();
        for (int k = (int)coeff.size() - 2; k >= 0; --k)
            b = b * z + coeff[k];
        return b;
    }

    // Same Horner, but also reports the largest intermediate |b_k|. The
    // round-off in the final result is on the order of eps_machine * maxPartial
    // (cancellation amplifies error by exactly this ratio of partial-to-final
    // magnitude). This is the quantity rescaling Horner exists to control, and
    // here we use it to know when arg(P) is no longer trustworthy.
    cd eval(cd z, double& maxPartial) const {
        cd b = coeff.back();
        maxPartial = std::abs(b);
        for (int k = (int)coeff.size() - 2; k >= 0; --k) {
            b = b * z + coeff[k];
            maxPartial = std::max(maxPartial, std::abs(b));
        }
        return b;
    }

    // P'(z), used by the Newton polishing pass.
    Polynomial derivative() const {
        Polynomial d;
        if (degree() < 1) { d.coeff = { cd(0,0) }; return d; }
        d.coeff.resize(coeff.size() - 1);
        for (int k = 1; k < (int)coeff.size(); ++k)
            d.coeff[k - 1] = coeff[k] * double(k);
        return d;
    }
};

// ===========================================================================
// Root-enclosing bound
// ===========================================================================
// Cauchy's bound: every root satisfies |z| <= 1 + max_{k<n} |c_k / c_n|.
// Simple but can be enormously loose (Wilkinson -> ~1e19), which both wastes
// subdivision levels and pushes the initial contour to a radius where Horner
// overflows.
double cauchy_bound(const Polynomial& P) {
    int n = P.degree();
    double cn = std::abs(P.coeff[n]);
    double m = 0.0;
    for (int k = 0; k < n; ++k)
        m = std::max(m, std::abs(P.coeff[k]) / cn);
    return 1.0 + m;
}

// Fujiwara's bound: |z| <= 2 * max_k |c_{n-k} / c_n|^{1/k}, with the k=n term
// halved. Much tighter than Cauchy on polynomials like Wilkinson (~25 vs 1e19),
// which keeps the contour in a range where double-precision Horner is safe.
double fujiwara_bound(const Polynomial& P) {
    int n = P.degree();
    double cn = std::abs(P.coeff[n]);
    double m = 0.0;
    for (int k = 1; k <= n; ++k) {
        double ratio = std::abs(P.coeff[n - k]) / cn;
        if (k == n) ratio *= 0.5;
        m = std::max(m, std::pow(ratio, 1.0 / k));
    }
    return 2.0 * m;
}

// Both are valid upper bounds, so their min is still valid -- take the tighter.
double root_bound(const Polynomial& P) {
    return std::min(cauchy_bound(P), fujiwara_bound(P));
}

// ===========================================================================
// Square + winding-number root count
// ===========================================================================
struct Square {
    cd     center;
    double half;        // half-width (L-infinity radius)
    int    count;       // root count, filled in by count_roots()
    bool   trusted;     // was the winding count above the noise floor?
};

// Count zeros of P strictly inside `sq` via the argument principle:
//   N = (1 / 2pi) * (total change in arg P(z) as z traverses the boundary CCW).
// `residual` returns |winding - round(winding)| as a confidence measure; for a
// clean contour it should be ~0. samplesPerSide controls contour resolution.
int winding_count(const Polynomial& P, const Square& sq,
                  int samplesPerSide, double& residual, bool& trustworthy) {
    const int S = samplesPerSide * 4;
    const double h = sq.half;
    const cd c = sq.center;

    // i in [0,S) -> a point on the boundary, CCW starting at the bottom-left
    // corner (bottom L->R, right B->T, top R->L, left T->B). CCW orientation is
    // what makes interior zeros count with a +sign.
    auto perim = [&](int i) -> cd {
        int side  = i / samplesPerSide;                       // 0..3
        double t  = double(i % samplesPerSide) / samplesPerSide; // [0,1)
        double x, y;
        switch (side) {
            case 0: x = -h + 2*h*t; y = -h;          break; // bottom
            case 1: x =  h;          y = -h + 2*h*t; break; // right
            case 2: x =  h - 2*h*t;  y =  h;         break; // top
            default:x = -h;          y =  h - 2*h*t; break; // left
        }
        return c + cd(x, y);
    };

    double total   = 0.0;
    double maxAbsP = 0.0;     // strongest signal anywhere on the contour
    double scaleP  = 0.0;     // largest Horner partial seen -> sets noise floor
    double mp;
    cd p0 = P.eval(perim(0), mp);
    maxAbsP = std::max(maxAbsP, std::abs(p0));
    scaleP  = std::max(scaleP, mp);
    double prevArg = std::arg(p0);

    for (int i = 1; i <= S; ++i) {            // i == S closes the loop (wraps to 0)
        cd p = P.eval(perim(i % S), mp);
        maxAbsP = std::max(maxAbsP, std::abs(p));
        scaleP  = std::max(scaleP, mp);
        double curArg = std::arg(p);
        double d = curArg - prevArg;
        while (d <= -PI) d += TWO_PI;          // unwrap into (-pi, pi]
        while (d >   PI) d -= TWO_PI;
        total  += d;
        prevArg = curArg;
    }
    double winding = total / TWO_PI;
    int    N       = (int)std::lround(winding);
    residual       = std::fabs(winding - N);

    // The winding integral tolerates a few noisy samples where the contour
    // grazes a root (normal during subdivision). It fails only when the ENTIRE
    // contour is at the rounding-noise floor -- i.e. even the strongest signal
    // on the boundary is comparable to eps_machine * (largest Horner partial).
    // That is the signature of having burrowed inside a multiple root.
    const double NOISE = 1e3 * 2.220446e-16;  // ~1e3 * eps_machine
    trustworthy = (maxAbsP > NOISE * scaleP);
    return N;
}

// Wrapper with adaptive resampling: if the winding number is suspiciously far
// from an integer (a contour passing close to a root, or under-sampling), add
// more samples and retry before trusting the count.
int count_roots(const Polynomial& P, Square& sq) {
    int sps = std::max(8, 4 * P.degree());     // ~4n samples per side to start
    double residual; bool trusted;
    int N = winding_count(P, sq, sps, residual, trusted);
    for (int tries = 0; residual > 0.1 && tries < 4; ++tries) {
        sps *= 2;
        N = winding_count(P, sq, sps, residual, trusted);
    }
    sq.count   = N;
    sq.trusted = trusted;
    return N;
}

// ===========================================================================
// Newton polishing (Phase-4 preview)
// ===========================================================================
// Modified Newton: z <- z - m * P/P'. The factor m = multiplicity (which we
// already know from the winding count!) restores quadratic convergence even at
// repeated roots, where plain Newton would crawl in linearly.
cd newton_polish(const Polynomial& P, const Polynomial& dP, cd z,
                 int mult, int iters = 12) {
    for (int i = 0; i < iters; ++i) {
        cd p = P.eval(z), d = dP.eval(z);
        if (std::abs(d) < 1e-300) break;
        cd step = double(mult) * (p / d);
        z -= step;
        if (std::abs(step) < 1e-15) break;
    }
    return z;
}

// ===========================================================================
// Solver
// ===========================================================================
struct FoundRoot { cd z; int mult; bool lowConfidence; };

std::vector<FoundRoot> find_roots(const Polynomial& P, double eps,
                                  bool polish = true) {
    std::vector<FoundRoot> roots;
    Polynomial dP = P.derivative();

    double R = root_bound(P);

    // The argument principle is undefined for a root that lands exactly on the
    // contour. Nudging the initial square by a small irrational-ish offset (and
    // enlarging it to still contain the disk |z|<=R) makes it vanishingly
    // unlikely that any root coincides with a dyadic subdivision grid line.
    cd offset = R * 1e-3 * cd(0.41421356, 0.31415926); // ~ (sqrt2-1, pi-3)
    double half = R * 1.05 + std::abs(offset);
    Square start{ offset, half, 0, true };
    count_roots(P, start);

    std::vector<Square> work = { start };
    while (!work.empty()) {
        std::vector<Square> next;
        for (auto& sq : work) {
            if (sq.count <= 0) continue;           // no roots here -> discard

            // Stop subdividing if the box is small enough OR if the contour has
            // sunk to the numeric noise floor (subdividing further only burrows
            // deeper into garbage -- this is what was spawning phantom roots at
            // repeated roots, and what flags Wilkinson as un-certifiable here).
            bool converged = (sq.half <= eps);
            if (converged || !sq.trusted) {
                cd z = sq.center;
                // Polish only when P can be evaluated accurately on this box.
                // For a noise-floor box (a multiple root) the subdivision center
                // is already at the theoretical accuracy limit ~ eps^(1/mult);
                // Newton there only chases noise and drifts.
                if (polish && sq.trusted) z = newton_polish(P, dP, z, sq.count);
                roots.push_back({ z, sq.count, !sq.trusted });
                continue;
            }

            // subdivide into 4 children, count each
            double hh = sq.half * 0.5;
            cd cc = sq.center;
            Square kids[4] = {
                { cc + cd(-hh,-hh), hh, 0, true },
                { cc + cd( hh,-hh), hh, 0, true },
                { cc + cd( hh, hh), hh, 0, true },
                { cc + cd(-hh, hh), hh, 0, true },
            };
            int sum = 0;
            for (auto& k : kids) sum += count_roots(P, k);

            // INVARIANT: children counts must sum to the parent count. This is
            // the single most useful debugging check in the whole algorithm.
            // Only meaningful when every box involved was trustworthy.
            bool allTrusted = sq.trusted;
            for (auto& k : kids) allTrusted = allTrusted && k.trusted;
            if (allTrusted && sum != sq.count) {
                std::fprintf(stderr,
                    "[warn] count mismatch: parent=%d  children=%d  "
                    "(center=%.4g%+.4gi half=%.4g)\n",
                    sq.count, sum, cc.real(), cc.imag(), sq.half);
            }
            for (auto& k : kids) if (k.count > 0) next.push_back(k);
        }
        work.swap(next);
    }
    return roots;
}

// ===========================================================================
// Test utilities
// ===========================================================================
// Build P(z) = prod (z - r_i) by repeated multiplication.
Polynomial from_roots(const std::vector<cd>& r) {
    Polynomial P; P.coeff = { cd(1,0) };
    for (cd root : r) {
        std::vector<cd> nc(P.coeff.size() + 1, cd(0,0));
        for (size_t k = 0; k < P.coeff.size(); ++k) {
            nc[k + 1] += P.coeff[k];           // * z
            nc[k]     += P.coeff[k] * (-root); // * (-root)
        }
        P.coeff = nc;
    }
    return P;
}

// Greedy nearest-neighbour match of found roots (expanded by multiplicity)
// against the known roots; returns the max matching error.
double max_error(const std::vector<cd>& truth,
                 const std::vector<FoundRoot>& found, int& totalMult) {
    std::vector<cd> est;
    for (auto& f : found)
        for (int i = 0; i < f.mult; ++i) est.push_back(f.z);
    totalMult = (int)est.size();

    std::vector<bool> used(est.size(), false);
    double maxerr = 0.0;
    for (cd tr : truth) {
        double best = 1e300; int bi = -1;
        for (size_t j = 0; j < est.size(); ++j) if (!used[j]) {
            double d = std::abs(tr - est[j]);
            if (d < best) { best = d; bi = (int)j; }
        }
        if (bi >= 0) { used[bi] = true; maxerr = std::max(maxerr, best); }
        else         { maxerr = 1e300; }       // a true root went unmatched
    }
    return maxerr;
}

void run_test(const std::string& name, const Polynomial& P,
              const std::vector<cd>& truth, double eps = 1e-8) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto found = find_roots(P, eps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int totalMult;
    double err = max_error(truth, found, totalMult);
    int lowConf = 0;
    for (auto& f : found) if (f.lowConfidence) lowConf++;

    std::printf("%-28s n=%-4d bound=%-10.4g squares=%-4zu "
                "roots(w/mult)=%-4d  max_err=%.3e  %.1f ms%s\n",
                name.c_str(), P.degree(), root_bound(P),
                found.size(), totalMult, err, ms,
                lowConf ? "  [LOW-CONFIDENCE boxes: numeric noise floor]" : "");
}

int main() {
    std::printf("Argument-principle subdivision root-finder -- CPU reference\n");
    std::printf("------------------------------------------------------------\n");

    // 1. (z-1)(z-2)(z-3)
    {
        std::vector<cd> r = { 1, 2, 3 };
        run_test("(z-1)(z-2)(z-3)", from_roots(r), r);
    }

    // 2. n-th roots of unity: z^n - 1
    for (int n : { 5, 12, 30 }) {
        Polynomial P; P.coeff.assign(n + 1, cd(0,0));
        P.coeff[0] = cd(-1,0); P.coeff[n] = cd(1,0);
        std::vector<cd> r;
        for (int k = 0; k < n; ++k)
            r.push_back(std::polar(1.0, TWO_PI * k / n));
        run_test("z^" + std::to_string(n) + " - 1", P, r);
    }

    // 3. Wilkinson's polynomial: prod_{k=1}^{20} (z - k)
    {
        std::vector<cd> r;
        for (int k = 1; k <= 20; ++k) r.push_back(cd(k, 0));
        run_test("Wilkinson prod (z-k), k=1..20", from_roots(r), r);
    }

    // 4. Repeated roots: (z-1)^3 (z+2)^2  -> tests multiplicity counting
    {
        std::vector<cd> r = { 1, 1, 1, -2, -2 };
        run_test("(z-1)^3 (z+2)^2", from_roots(r), r);
    }

    // 5. Complex + clustered roots
    {
        std::vector<cd> r = {
            cd(0.0, 1.0), cd(0.0, -1.0),          // +-i
            cd(2.0, 0.0),
            cd(0.5, 0.5), cd(0.5001, 0.5),        // a tight cluster
            cd(-3.0, 2.0)
        };
        run_test("complex + clustered", from_roots(r), r);
    }

    return 0;
}
