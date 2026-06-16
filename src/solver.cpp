// ---------------------------------------------------------------------------
// Top-level subdivision solver implementation.
// ---------------------------------------------------------------------------
#include "polyroots/solver.hpp"
#include "polyroots/newton.hpp"
#include "polyroots/root_bound.hpp"
#include "polyroots/winding.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

namespace polyroots {

std::vector<FoundRoot> find_roots(const Polynomial& P, double eps, bool polish) {
    std::vector<FoundRoot> roots;
    Polynomial dP = P.derivative();

    double R = root_bound(P);

    // The argument principle is undefined for a root that lands exactly on the
    // contour. Nudging the initial square by a small irrational-ish offset (and
    // enlarging it to still contain the disk |z| <= R) makes it vanishingly
    // unlikely that any root coincides with a dyadic subdivision grid line.
    cd offset = R * 1e-3 * cd(0.41421356, 0.31415926); // ~ (sqrt2-1, pi/10)
    double half = R * 1.05 + std::abs(offset);
    Square start{ offset, half, 0, true };
    count_roots(P, start);

    std::vector<Square> work = { start };
    while (!work.empty()) {
        std::vector<Square> next;
        for (auto& sq : work) {
            if (sq.count <= 0) continue; // no roots here -> discard

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
                if (polish && sq.trusted)
                    z = newton_polish(P, dP, z, sq.count);
                roots.push_back({ z, sq.count, !sq.trusted });
                continue;
            }

            // subdivide into 4 children, count each
            double hh = sq.half * 0.5;
            cd     cc = sq.center;
            Square kids[4] = {
                { cc + cd(-hh, -hh), hh, 0, true },
                { cc + cd( hh, -hh), hh, 0, true },
                { cc + cd( hh,  hh), hh, 0, true },
                { cc + cd(-hh,  hh), hh, 0, true },
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
            for (auto& k : kids)
                if (k.count > 0) next.push_back(k);
        }
        work.swap(next);
    }
    return roots;
}

} // namespace polyroots
