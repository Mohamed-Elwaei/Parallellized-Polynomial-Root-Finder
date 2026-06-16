// ---------------------------------------------------------------------------
// Polynomial implementation.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"
#include <algorithm>
#include <cmath>

namespace polyroots {

cd Polynomial::eval(cd z) const {
    cd b = coeff.back();
    for (int k = static_cast<int>(coeff.size()) - 2; k >= 0; --k)
        b = b * z + coeff[k];
    return b;
}

cd Polynomial::eval(cd z, double& maxPartial) const {
    cd b = coeff.back();
    maxPartial = std::abs(b);
    for (int k = static_cast<int>(coeff.size()) - 2; k >= 0; --k) {
        b = b * z + coeff[k];
        maxPartial = std::max(maxPartial, std::abs(b));
    }
    return b;
}

Polynomial Polynomial::derivative() const {
    Polynomial d;
    if (degree() < 1) { d.coeff = { cd(0, 0) }; return d; }
    d.coeff.resize(coeff.size() - 1);
    for (int k = 1; k < static_cast<int>(coeff.size()); ++k)
        d.coeff[k - 1] = coeff[k] * static_cast<double>(k);
    return d;
}

Polynomial from_roots(const std::vector<cd>& roots) {
    Polynomial P;
    P.coeff = { cd(1, 0) };
    for (cd root : roots) {
        std::vector<cd> nc(P.coeff.size() + 1, cd(0, 0));
        for (std::size_t k = 0; k < P.coeff.size(); ++k) {
            nc[k + 1] += P.coeff[k];            // multiply by z
            nc[k]     += P.coeff[k] * (-root);  // multiply by (-root)
        }
        P.coeff = nc;
    }
    return P;
}

} // namespace polyroots
