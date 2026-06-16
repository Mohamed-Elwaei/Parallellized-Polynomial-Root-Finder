// ---------------------------------------------------------------------------
// demo.cpp -- exercises the polyroots library on a battery of polynomials and
// prints a one-line summary per case. This is the human-facing smoke test;
// the machine-checkable assertions live in tests/.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"
#include "polyroots/root_bound.hpp"
#include "polyroots/solver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace polyroots;

// Greedy nearest-neighbour match of found roots (expanded by multiplicity)
// against the known roots; returns the max matching error.
static double max_error(const std::vector<cd>& truth,
                        const std::vector<FoundRoot>& found, int& totalMult) {
    std::vector<cd> est;
    for (auto& f : found)
        for (int i = 0; i < f.mult; ++i) est.push_back(f.z);
    totalMult = static_cast<int>(est.size());

    std::vector<bool> used(est.size(), false);
    double maxerr = 0.0;
    for (cd tr : truth) {
        double best = 1e300;
        int    bi   = -1;
        for (std::size_t j = 0; j < est.size(); ++j)
            if (!used[j]) {
                double d = std::abs(tr - est[j]);
                if (d < best) { best = d; bi = static_cast<int>(j); }
            }
        if (bi >= 0) { used[bi] = true; maxerr = std::max(maxerr, best); }
        else         { maxerr = 1e300; } // a true root went unmatched
    }
    return maxerr;
}

static void run_test(const std::string& name, const Polynomial& P,
                     const std::vector<cd>& truth, double eps = 1e-8) {
    auto   t0    = std::chrono::high_resolution_clock::now();
    auto   found = find_roots(P, eps);
    auto   t1    = std::chrono::high_resolution_clock::now();
    double ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int    totalMult;
    double err     = max_error(truth, found, totalMult);
    int    lowConf = 0;
    for (auto& f : found)
        if (f.lowConfidence) lowConf++;

    std::printf("%-28s n=%-4d bound=%-10.4g squares=%-4zu "
                "roots(w/mult)=%-4d  max_err=%.3e  %.1f ms%s\n",
                name.c_str(), P.degree(), root_bound(P), found.size(),
                totalMult, err, ms,
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
        Polynomial P;
        P.coeff.assign(n + 1, cd(0, 0));
        P.coeff[0] = cd(-1, 0);
        P.coeff[n] = cd(1, 0);
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
            cd(0.0, 1.0), cd(0.0, -1.0),   // +-i
            cd(2.0, 0.0),
            cd(0.5, 0.5), cd(0.5001, 0.5), // a tight cluster
            cd(-3.0, 2.0)
        };
        run_test("complex + clustered", from_roots(r), r);
    }

    return 0;
}
