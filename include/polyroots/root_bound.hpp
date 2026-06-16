#pragma once
// ---------------------------------------------------------------------------
// Root-enclosing bounds: every root of P satisfies |z| <= bound.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"

namespace polyroots {

// Cauchy's bound: |z| <= 1 + max_{k<n} |c_k / c_n|. Simple but can be
// enormously loose (Wilkinson -> ~1e19).
double cauchy_bound(const Polynomial& P);

// Fujiwara's bound: |z| <= 2 * max_k |c_{n-k}/c_n|^{1/k}, with the k=n term
// halved. Much tighter on polynomials like Wilkinson (~25 vs 1e19).
double fujiwara_bound(const Polynomial& P);

// Both are valid upper bounds, so their min is still valid -- take the tighter.
double root_bound(const Polynomial& P);

} // namespace polyroots
