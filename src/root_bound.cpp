// ---------------------------------------------------------------------------
// Root-enclosing bounds implementation.
// ---------------------------------------------------------------------------
#include "polyroots/root_bound.hpp"
#include <algorithm>
#include <cmath>

namespace polyroots {

double cauchy_bound(const Polynomial& P) {
    int n = P.degree();
    double cn = std::abs(P.coeff[n]);
    double m = 0.0;
    for (int k = 0; k < n; ++k)
        m = std::max(m, std::abs(P.coeff[k]) / cn);
    return 1.0 + m;
}

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

double root_bound(const Polynomial& P) {
    return std::min(cauchy_bound(P), fujiwara_bound(P));
}

} // namespace polyroots
