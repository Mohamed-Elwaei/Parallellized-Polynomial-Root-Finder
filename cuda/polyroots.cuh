// ===========================================================================
// polyroots CUDA -- shared host/device math for the prototypes (P0, P1, P2).
// ---------------------------------------------------------------------------
// Everything here is __host__ __device__ and `inline`, so the SAME code runs on
// the CPU and the GPU and can be included from multiple .cu files without
// multiple-definition errors. The CPU stays a faithful oracle by construction.
//
// On Kaggle, write this to a file first, then the .cu files include it:
//   %%writefile polyroots.cuh
//   <paste this>
// and each prototype begins with:  #include "polyroots.cuh"
// ===========================================================================
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <thrust/complex.h>

using cmplx = thrust::complex<double>;

// Check every CUDA call -- most "mysterious" GPU bugs are just unchecked errors.
#define CK(x) do { cudaError_t e_ = (x);                                       \
    if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s:%d: %s\n",    \
        __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// constexpr is a compile-time constant: usable on host AND device, no qualifier.
constexpr double kPI = 3.14159265358979323846;

// A square cell: centre (cx,cy) and half-width.
struct Cell { double cx, cy, half; };

// Wrap a small arg-difference into (-pi, pi]. Because arg() returns values in an
// interval of width 2pi, the raw difference d = cur - prev lies in [-2pi, 2pi],
// so a SINGLE +-2pi correction always suffices -- no loop needed.
__host__ __device__ inline double unwrap(double d) {
    if      (d <= -kPI) d += 2 * kPI;
    else if (d >   kPI) d -= 2 * kPI;
    return d;
}

// Horner evaluation. coeff ascending: coeff[k] is the coefficient of z^k.
__host__ __device__ inline
cmplx poly_eval(const cmplx* coeff, int ncoeff, cmplx z) {
    cmplx b = coeff[ncoeff - 1];
    for (int k = ncoeff - 2; k >= 0; --k) b = b * z + coeff[k];
    return b;
}

// Point on the boundary of the square [center +- half], CCW from the bottom-left
// corner (bottom L->R, right B->T, top R->L, left T->B). i in [0, 4*sps).
__host__ __device__ inline
cmplx perim_point(cmplx center, double half, int sps, int i) {
    int side = i / sps; double t = double(i % sps) / sps, x, y;
    if      (side == 0) { x = -half + 2 * half * t; y = -half; }
    else if (side == 1) { x =  half;                y = -half + 2 * half * t; }
    else if (side == 2) { x =  half - 2 * half * t; y =  half; }
    else                { x = -half;                y =  half - 2 * half * t; }
    return center + cmplx(x, y);
}

// Winding-number root count inside the square: (1/2pi) * total change in
// arg P(z) around the boundary CCW, rounded to an integer.
__host__ __device__ inline
int winding_count(const cmplx* coeff, int ncoeff, cmplx center, double half, int sps) {
    const int S = sps * 4;
    cmplx  p0    = poly_eval(coeff, ncoeff, perim_point(center, half, sps, 0));
    double prev  = thrust::arg(p0);
    double total = 0.0;
    for (int i = 1; i <= S; ++i) {
        cmplx  p   = poly_eval(coeff, ncoeff, perim_point(center, half, sps, i % S));
        double cur = thrust::arg(p);
        total += unwrap(cur - prev);
        prev   = cur;
    }
    double w = total / (2 * kPI);
    return (int)(w >= 0 ? w + 0.5 : w - 0.5);
}

// Accumulated unwrapped change in arg(P) along the segment A->B over S
// sub-intervals, endpoints included. Returns the REAL phase change (radians).
__host__ __device__ inline
double edge_phase(const cmplx* coeff, int ncoeff, cmplx A, cmplx B, int S) {
    cmplx  p     = poly_eval(coeff, ncoeff, A);
    double prev  = thrust::arg(p);
    double total = 0.0;
    for (int k = 1; k <= S; ++k) {
        cmplx  q   = poly_eval(coeff, ncoeff, A + (double(k) / S) * (B - A));
        double cur = thrust::arg(q);
        total += unwrap(cur - prev);
        prev   = cur;
    }
    return total;
}

// Grid-line coordinates for an N x N grid of cells, half-width h, origin offset.
__host__ __device__ inline double gx(double ox, double h, int N, int col) { return ox + (2 * col - N) * h; }
__host__ __device__ inline double gy(double oy, double h, int N, int row) { return oy + (2 * row - N) * h; }

// Plain Newton (multiplicity 1; an isolated cell has a single root).
__host__ __device__ inline
cmplx newton_polish(const cmplx* coeff, int nc, const cmplx* dcoeff, int dn, cmplx z) {
    for (int i = 0; i < 60; ++i) {
        cmplx p = poly_eval(coeff, nc, z), d = poly_eval(dcoeff, dn, z);
        if (thrust::abs(d) < 1e-300) break;
        cmplx step = p / d; z -= step;
        // RELATIVE convergence test. An absolute `< 1e-15` is scale-dependent:
        // for roots of magnitude ~1e-9 it stops at only ~1e-6 relative accuracy,
        // and for magnitude ~1e9 it never triggers. Comparing against |z| makes
        // the accuracy achieved independent of the problem's scale. (A root at
        // exactly 0 gives a zero tolerance and simply runs the iteration cap,
        // which is harmless -- quadratic convergence gets there long before.)
        if (thrust::abs(step) <= 1e-15 * thrust::abs(z)) break;
    }
    return z;
}
