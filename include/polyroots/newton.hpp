#pragma once
// ---------------------------------------------------------------------------
// Modified Newton polishing.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"

namespace polyroots {

// Modified Newton: z <- z - mult * P/P'. The factor mult = multiplicity (known
// from the winding count) restores quadratic convergence even at repeated
// roots, where plain Newton would converge only linearly.
cd newton_polish(const Polynomial& P, const Polynomial& dP, cd z,
                 int mult, int iters = 12);

} // namespace polyroots
