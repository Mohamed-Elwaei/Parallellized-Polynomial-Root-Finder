#pragma once
// ---------------------------------------------------------------------------
// Polynomial: dense complex polynomial with Horner evaluation.
//
// Coefficients are stored ascending: coeff[k] is the coefficient of z^k, so
// coeff.back() is the leading term c_n. This ordering is what the rescaling-
// Horner invariant in the roadmap relies on.
// ---------------------------------------------------------------------------
#include "polyroots/constants.hpp"
#include <vector>

namespace polyroots {

class Polynomial {
public:
    std::vector<cd> coeff; // coeff[k] = coefficient of z^k; coeff.back() = c_n

    Polynomial() = default;
    explicit Polynomial(std::vector<cd> c) : coeff(std::move(c)) {}

    // Degree n = size - 1. A single-coefficient polynomial is degree 0.
    int degree() const { return static_cast<int>(coeff.size()) - 1; }

    // Plain Horner evaluation.
    cd eval(cd z) const;

    // Horner evaluation that also reports the largest intermediate |b_k|.
    // The round-off in the result is ~ eps_machine * maxPartial: cancellation
    // amplifies error by exactly the partial-to-final magnitude ratio. The
    // winding routine uses this to know when arg(P) is no longer trustworthy.
    cd eval(cd z, double& maxPartial) const;

    // Formal derivative P'(z), used by the Newton polishing pass.
    Polynomial derivative() const;
};

// Build P(z) = prod_i (z - roots[i]) by repeated multiplication. Handy for
// constructing test polynomials with known roots.
Polynomial from_roots(const std::vector<cd>& roots);

} // namespace polyroots
