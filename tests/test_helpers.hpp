#pragma once
// ---------------------------------------------------------------------------
// Test helpers shared across the test translation units.
// ---------------------------------------------------------------------------
#include "polyroots/solver.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace polyroots::test {

// Greedy nearest-neighbour match of found roots (expanded by multiplicity)
// against known roots; returns the worst per-root error. Returns a huge value
// if any true root goes unmatched (count too low).
inline double max_error(const std::vector<cd>& truth,
                        const std::vector<FoundRoot>& found) {
    std::vector<cd> est;
    for (const auto& f : found)
        for (int i = 0; i < f.mult; ++i) est.push_back(f.z);

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
        else         { maxerr = 1e300; }
    }
    return maxerr;
}

// Total root count with multiplicity.
inline int total_multiplicity(const std::vector<FoundRoot>& found) {
    int s = 0;
    for (const auto& f : found) s += f.mult;
    return s;
}

} // namespace polyroots::test
