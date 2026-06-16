// ---------------------------------------------------------------------------
// Winding-number root counting implementation.
// ---------------------------------------------------------------------------
#include "polyroots/winding.hpp"
#include <algorithm>
#include <cmath>

namespace polyroots {

int winding_count(const Polynomial& P, const Square& sq,
                  int samplesPerSide, double& residual, bool& trustworthy) {
    const int    S = samplesPerSide * 4;
    const double h = sq.half;
    const cd     c = sq.center;

    // i in [0,S) -> a point on the boundary, CCW starting at the bottom-left
    // corner (bottom L->R, right B->T, top R->L, left T->B). CCW orientation is
    // what makes interior zeros count with a + sign.
    auto perim = [&](int i) -> cd {
        int    side = i / samplesPerSide;                          // 0..3
        double t    = double(i % samplesPerSide) / samplesPerSide; // [0,1)
        double x, y;
        switch (side) {
            case 0:  x = -h + 2 * h * t; y = -h;            break; // bottom
            case 1:  x =  h;             y = -h + 2 * h * t; break; // right
            case 2:  x =  h - 2 * h * t; y =  h;            break; // top
            default: x = -h;             y =  h - 2 * h * t; break; // left
        }
        return c + cd(x, y);
    };

    double total   = 0.0;
    double maxAbsP = 0.0; // strongest signal anywhere on the contour
    double scaleP  = 0.0; // largest Horner partial seen -> sets the noise floor
    double mp;
    cd p0 = P.eval(perim(0), mp);
    maxAbsP = std::max(maxAbsP, std::abs(p0));
    scaleP  = std::max(scaleP, mp);
    double prevArg = std::arg(p0);

    for (int i = 1; i <= S; ++i) {     // i == S closes the loop (wraps to 0)
        cd p = P.eval(perim(i % S), mp);
        maxAbsP = std::max(maxAbsP, std::abs(p));
        scaleP  = std::max(scaleP, mp);
        double curArg = std::arg(p);
        double d = curArg - prevArg;
        while (d <= -PI) d += TWO_PI;  // unwrap into (-pi, pi]
        while (d >   PI) d -= TWO_PI;
        total  += d;
        prevArg = curArg;
    }
    double winding = total / TWO_PI;
    int    N       = static_cast<int>(std::lround(winding));
    residual       = std::fabs(winding - N);

    // The winding integral tolerates a few noisy samples where the contour
    // grazes a root (normal during subdivision). It fails only when the ENTIRE
    // contour is at the rounding-noise floor -- i.e. even the strongest signal
    // on the boundary is comparable to eps_machine * (largest Horner partial).
    // That is the signature of having burrowed inside a multiple root.
    const double NOISE = 1e3 * EPS_MACHINE;
    trustworthy = (maxAbsP > NOISE * scaleP);
    return N;
}

int count_roots(const Polynomial& P, Square& sq) {
    int sps = std::max(8, 4 * P.degree()); // ~4n samples per side to start
    double residual;
    bool   trusted;
    int N = winding_count(P, sq, sps, residual, trusted);
    for (int tries = 0; residual > 0.1 && tries < 4; ++tries) {
        sps *= 2;
        N = winding_count(P, sq, sps, residual, trusted);
    }
    sq.count   = (N > 0) ? N : 0;
    sq.trusted = trusted;
    return sq.count;
}

} // namespace polyroots
