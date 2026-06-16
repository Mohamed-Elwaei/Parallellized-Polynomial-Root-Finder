#pragma once
// ---------------------------------------------------------------------------
// Shared numeric constants for the polynomial root finder.
// ---------------------------------------------------------------------------
#include <complex>

namespace polyroots {

using cd = std::complex<double>;

inline constexpr double PI     = 3.14159265358979323846;
inline constexpr double TWO_PI = 2.0 * PI;

// IEEE-754 double machine epsilon (2^-52).
inline constexpr double EPS_MACHINE = 2.220446049250313e-16;

} // namespace polyroots
