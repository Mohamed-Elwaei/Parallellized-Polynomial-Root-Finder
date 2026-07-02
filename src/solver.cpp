// ---------------------------------------------------------------------------
// Top-level subdivision solver implementation.
// ---------------------------------------------------------------------------
#include "polyroots/solver.hpp"
#include "polyroots/newton.hpp"
#include "polyroots/root_bound.hpp"
#include "polyroots/winding.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
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

    // A healthy run subdivides only ~24*degree boxes overall (measured). A
    // multiple/clustered root, by contrast, sinks to the rounding-noise floor
    // and can spawn phantom boxes that multiply ~4x per level -- the unbounded
    // growth that used to hang the solver. 2000 + 100*degree leaves >4x headroom
    // over every well-conditioned case while still bailing out of a phantom
    // runaway in ~a second or two. (A clean fix for the phantom case itself is
    // Phase-2 -- see the note in the subdivision loop.)
    const std::size_t maxBoxes =
        2000 + 100 * static_cast<std::size_t>(std::max(1, P.degree()));
    std::size_t processed = 0;

    std::vector<Square> work = { start };
    while (!work.empty()) {
        std::vector<Square> next;
        for (auto& sq : work) {
            if (sq.count <= 0) continue; // no roots here -> discard

            if (++processed > maxBoxes) {
                // Safety net: emit what we have as low-confidence and stop.
                roots.push_back({ sq.center, sq.count, true });
                next.clear();
                break;
            }

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
            for (auto& k : kids) count_roots(P, k);

            // NOTE: in a clean run the children's counts sum to the parent's.
            // A mismatch is the signature of having sunk to the rounding-noise
            // floor inside a tight/multiple-root cluster. We do NOT act on it
            // here: a robust response (distinguishing a true multiple root from
            // separable roots that merely graze a subdivision edge) requires the
            // Phase-2 certified-disk + compensated-Horner machinery. Until then,
            // such boxes are left to terminate via the noise-floor (!trusted)
            // path above, and the work-list cap below guarantees termination.
            for (auto& k : kids)
                if (k.count > 0) next.push_back(k);
        }
        work.swap(next);
    }
    return roots;
}

} // namespace polyroots
