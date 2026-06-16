#pragma once
// ---------------------------------------------------------------------------
// Top-level solver: subdivision root finder driven by the argument principle.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"
#include <vector>

namespace polyroots {

// A located root, its multiplicity, and whether it was found on a noise-floor
// (low-confidence) box rather than a cleanly converged one.
struct FoundRoot {
    cd   z;
    int  mult;
    bool lowConfidence;
};

// Find all roots of P to absolute tolerance `eps` (in the L-infinity box
// metric). When `polish` is true, trusted boxes are refined with modified
// Newton. Roots are returned with multiplicity collapsed into FoundRoot::mult.
std::vector<FoundRoot> find_roots(const Polynomial& P, double eps,
                                  bool polish = true);

} // namespace polyroots
