#pragma once
// ---------------------------------------------------------------------------
// Winding-number root counting via the argument principle.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"

namespace polyroots {

// An axis-aligned square region in the complex plane (L-infinity ball).
struct Square {
    cd     center;
    double half = 0.0;   // half-width (L-infinity radius)
    int    count = 0;    // root count, filled in by count_roots()
    bool   trusted = true; // was the winding count above the noise floor?
};

// Count zeros of P strictly inside `sq` via the argument principle:
//   N = (1 / 2pi) * (total change in arg P(z) as z traverses the boundary CCW).
// `residual` returns |winding - round(winding)| as a confidence measure; for a
// clean contour it is ~0. `trustworthy` is false when the whole contour has
// sunk to the rounding-noise floor (the signature of a multiple root).
int winding_count(const Polynomial& P, const Square& sq,
                  int samplesPerSide, double& residual, bool& trustworthy);

// Adaptive wrapper: if the winding number is suspiciously far from an integer,
// add more boundary samples and retry before trusting the count. Fills in
// sq.count and sq.trusted and returns sq.count.
int count_roots(const Polynomial& P, Square& sq);

} // namespace polyroots
