// ---------------------------------------------------------------------------
// Modified Newton polishing implementation.
// ---------------------------------------------------------------------------
#include "polyroots/newton.hpp"
#include <cmath>

namespace polyroots {

cd newton_polish(const Polynomial& P, const Polynomial& dP, cd z,
                 int mult, int iters) {
    for (int i = 0; i < iters; ++i) {
        cd p = P.eval(z), d = dP.eval(z);
        if (std::abs(d) < 1e-300) break;
        cd step = static_cast<double>(mult) * (p / d);
        z -= step;
        if (std::abs(step) < 1e-15) break;
    }
    return z;
}

} // namespace polyroots
