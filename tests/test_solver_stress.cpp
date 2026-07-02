// ---------------------------------------------------------------------------
// Stress / robustness suite for find_roots.
//
// Where test_solver.cpp is a hand-picked smoke battery (six known polynomials,
// degree <= 30, friendly coefficients), this file pushes the solver along the
// axes that actually break it:
//
//   Bucket A  Degree scaling + randomized round-trip   -- "does it stay correct
//             and terminate as n grows / over many random layouts?"
//   Bucket B  Degenerate & edge inputs                 -- constant, linear, root
//             at the origin, monomial z^n, real-axis roots.
//   Bucket C  Numerical extremes                       -- high multiplicity,
//             tight clusters, wide dynamic range, coefficient overflow.
//   Bucket D  Metamorphic invariants                   -- properties that must
//             hold for ANY polynomial (scale-invariance, conjugate symmetry).
//
// Oracle strategy (no external solver -- see README rationale):
//   * construction truth   : from_roots(known)  -> recover known
//   * closed form          : roots of unity
//   * residual             : |P(r)| ~ 0 at every returned root (independent of
//                            knowing the roots; catches false positives)
//   * metamorphic          : invariants that need no reference at all
//
// A genuinely independent gold reference would have to be MULTIPRECISION
// (e.g. MPSolve); a double-precision companion-matrix eigensolver shares this
// solver's cancellation failure modes and so is not independent on the hard
// cases. If that is ever wanted it belongs here as an optional CTest fixture.
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"
#include "polyroots/root_bound.hpp"
#include "polyroots/solver.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using namespace polyroots;
using polyroots::test::max_error;
using polyroots::test::total_multiplicity;

namespace {

// Largest |P(r)| over all returned roots, normalised by the coefficient scale
// so the threshold is dimensionless. This is the oracle that needs no knowledge
// of the true roots: a genuine root drives P to ~0; a phantom does not.
double max_relative_residual(const Polynomial& P,
                             const std::vector<FoundRoot>& found) {
    double cscale = 0.0;
    for (const cd& c : P.coeff) cscale = std::max(cscale, std::abs(c));
    if (cscale == 0.0) cscale = 1.0;

    double worst = 0.0;
    for (const auto& f : found)
        worst = std::max(worst, std::abs(P.eval(f.z)) / cscale);
    return worst;
}

// Random roots in the disk |z| <= radius with a guaranteed minimum pairwise
// separation, so a "well-conditioned" round-trip test does not accidentally
// manufacture a tight cluster (which is exercised deliberately elsewhere).
std::vector<cd> random_well_separated_roots(std::mt19937& rng, int n,
                                            double radius, double minSep) {
    std::uniform_real_distribution<double> U(-radius, radius);
    std::vector<cd> roots;
    int guard = 0;
    while (static_cast<int>(roots.size()) < n && guard++ < 100000) {
        cd cand(U(rng), U(rng));
        if (std::abs(cand) > radius) continue;
        bool ok = true;
        for (const cd& r : roots)
            if (std::abs(r - cand) < minSep) { ok = false; break; }
        if (ok) roots.push_back(cand);
    }
    return roots;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

} // namespace

// ===========================================================================
// Bucket A -- degree scaling + randomized round-trip
// ===========================================================================

// Roots of unity give exact known roots at arbitrary degree, so this isolates
// "does accuracy and runtime hold as n grows" from any construction error.
class SolverDegreeSweep : public ::testing::TestWithParam<int> {};

TEST_P(SolverDegreeSweep, RootsOfUnityRecovered) {
    const int n = GetParam();
    Polynomial P;
    P.coeff.assign(n + 1, cd(0, 0));
    P.coeff[0] = cd(-1, 0);
    P.coeff[n] = cd(1, 0);

    std::vector<cd> truth;
    for (int k = 0; k < n; ++k)
        truth.push_back(std::polar(1.0, TWO_PI * k / n));

    auto t0    = std::chrono::steady_clock::now();
    auto found = find_roots(P, 1e-8);
    double secs = seconds_since(t0);

    EXPECT_EQ(total_multiplicity(found), n) << "lost roots at degree " << n;
    // Accuracy ceiling loosens gently with degree; stay comfortably above it.
    EXPECT_LT(max_error(truth, found), 1e-5) << "degree " << n;
    EXPECT_LT(max_relative_residual(P, found), 1e-6) << "degree " << n;
    // Soft termination guard: a regression that explodes the work-list should
    // fail loudly here rather than hang CI. Generous so it is not flaky.
    EXPECT_LT(secs, 20.0) << "degree " << n << " took " << secs << "s";
}

INSTANTIATE_TEST_SUITE_P(Scaling, SolverDegreeSweep,
                         ::testing::Values(8, 16, 32, 64));

// Randomized round-trip over many seeds: replaces "I hope these few layouts are
// representative" with a distribution of well-conditioned cases.
class SolverRandomRoundTrip
    : public ::testing::TestWithParam<std::tuple<int, unsigned>> {};

TEST_P(SolverRandomRoundTrip, RecoversConstructedRoots) {
    const int      n    = std::get<0>(GetParam());
    const unsigned seed = std::get<1>(GetParam());
    std::mt19937   rng(seed);

    std::vector<cd> truth =
        random_well_separated_roots(rng, n, /*radius=*/1.5, /*minSep=*/0.25);
    ASSERT_EQ(static_cast<int>(truth.size()), n) << "seed " << seed;

    Polynomial P     = from_roots(truth);
    auto       found = find_roots(P, 1e-9);

    EXPECT_EQ(total_multiplicity(found), n)
        << "seed " << seed << " degree " << n;
    EXPECT_LT(max_error(truth, found), 1e-4)
        << "seed " << seed << " degree " << n;
    EXPECT_LT(max_relative_residual(P, found), 1e-6)
        << "seed " << seed << " degree " << n;
}

INSTANTIATE_TEST_SUITE_P(
    Randomized, SolverRandomRoundTrip,
    ::testing::Combine(::testing::Values(6, 12, 20),
                       ::testing::Values(1u, 2u, 3u, 4u, 5u)));

// ===========================================================================
// Bucket B -- degenerate & edge inputs
// ===========================================================================

TEST(SolverEdge, ConstantHasNoRoots) {
    Polynomial P({ cd(7, 0) }); // degree 0
    auto found = find_roots(P, 1e-8);
    EXPECT_EQ(total_multiplicity(found), 0);
}

TEST(SolverEdge, LinearSingleRoot) {
    // 2z - 6  ->  root at 3.
    Polynomial P({ cd(-6, 0), cd(2, 0) });
    auto found = find_roots(P, 1e-10);
    ASSERT_EQ(total_multiplicity(found), 1);
    EXPECT_NEAR(found.front().z.real(), 3.0, 1e-8);
    EXPECT_NEAR(found.front().z.imag(), 0.0, 1e-8);
}

TEST(SolverEdge, RootExactlyAtOrigin) {
    // z(z-1)(z-2): a root sits on the origin, near the solver's seed offset.
    std::vector<cd> truth = { cd(0, 0), cd(1, 0), cd(2, 0) };
    auto found = find_roots(from_roots(truth), 1e-9);
    EXPECT_EQ(total_multiplicity(found), 3);
    EXPECT_LT(max_error(truth, found), 1e-6);
}

TEST(SolverEdge, RootsOnTheRealAxis) {
    // All roots collinear on the real axis -- stresses the grid-offset nudge
    // that keeps roots off subdivision lines.
    std::vector<cd> truth = { cd(-5, 0), cd(-1, 0), cd(0.5, 0), cd(4, 0) };
    auto found = find_roots(from_roots(truth), 1e-9);
    EXPECT_EQ(total_multiplicity(found), 4);
    EXPECT_LT(max_error(truth, found), 1e-6);
    for (const auto& f : found) EXPECT_NEAR(f.z.imag(), 0.0, 1e-6);
}

// PINNED behaviour (not aspirational): the monomial z^n collapses Fujiwara's
// bound to 0, so the seed square has zero size and the n-fold root at the
// origin is never enclosed. This is a documented Phase-1 gap, mirroring the
// Wilkinson pin in test_solver.cpp; if a future change makes it recover the
// root, this assertion flips and the test should be updated.
TEST(SolverEdge, MonomialIsAKnownGap) {
    Polynomial P;
    P.coeff.assign(6, cd(0, 0));
    P.coeff[5] = cd(1, 0); // z^5
    EXPECT_DOUBLE_EQ(root_bound(P), 0.0) << "premise: bound collapses to 0";

    auto found = find_roots(P, 1e-8);
    EXPECT_EQ(total_multiplicity(found), 0)
        << "If this now finds the 5-fold root at 0, the seed-box setup "
           "improved -- update this test.";
}

// ===========================================================================
// Bucket C -- numerical extremes
// ===========================================================================

TEST(SolverExtreme, HighMultiplicity) {
    // (z-1)^8: position accuracy is capped at ~eps^(1/8), so assert the
    // MULTIPLICITY is right even though the location is coarse.
    std::vector<cd> truth(8, cd(1, 0));
    auto found = find_roots(from_roots(truth), 1e-8);
    EXPECT_EQ(total_multiplicity(found), 8);

    int multAtOne = 0;
    for (const auto& f : found)
        if (std::abs(f.z - cd(1, 0)) < 1e-1) multAtOne += f.mult;
    EXPECT_EQ(multAtOne, 8);
}

// TERMINATION GUARANTEE. For large m, (z-1)^m sinks to the rounding-noise floor
// and -- before the work-list cap was added -- spawned phantom boxes that
// multiplied ~4x per subdivision level, so the solver effectively HUNG (m=32 ran
// for minutes and m=48 never finished in practice). The cap now guarantees the
// solver returns, in bounded time, on any input.
//
// We deliberately do NOT assert a correct count or position here: cleanly
// resolving a high-multiplicity cluster (it currently reports a noise-corrupted
// count with phantom over/under-counting) is the documented Phase-2 work
// (certified-disk criterion + compensated Horner). What this pins is the safety
// property the cap buys: termination with finite coordinates, never a hang.
TEST(SolverExtreme, HighMultiplicityTerminates) {
    Polynomial P = from_roots(std::vector<cd>(32, cd(1, 0)));

    auto t0    = std::chrono::steady_clock::now();
    auto found = find_roots(P, 1e-6);
    double secs = seconds_since(t0);

    EXPECT_LT(secs, 15.0) << "did not terminate promptly (was an unbounded hang)";
    EXPECT_FALSE(found.empty());
    for (const auto& f : found) {
        EXPECT_TRUE(std::isfinite(f.z.real()));
        EXPECT_TRUE(std::isfinite(f.z.imag()));
    }
}

TEST(SolverExtreme, TightCluster) {
    // Two roots 1e-7 apart -- pushes subdivision depth and the noise-floor
    // trust heuristic. They may or may not separate, but the total count and
    // residual must hold.
    std::vector<cd> truth = {
        cd(0.3000000, 0.2), cd(0.3000001, 0.2), cd(-1.0, 0.0), cd(2.0, 1.0)
    };
    Polynomial P     = from_roots(truth);
    auto       found = find_roots(P, 1e-11);
    EXPECT_EQ(total_multiplicity(found), 4);
    EXPECT_LT(max_relative_residual(P, found), 1e-6);
}

// PINNED behaviour: roots spanning ~8 orders of magnitude. This is a documented
// Phase-1 gap in the SAME family as Wilkinson -- triggered by dynamic range
// rather than degree. Near the origin, Horner's partial sums reach ~|c_1 * z|
// (here ~1e4 * h) while |P| itself is ~1, so cancellation corrupts arg(P); the
// near-origin boxes read as empty and get pruned, and the small roots (1 and
// 1e-4) are silently lost. Only the large, well-conditioned root at 1e4
// survives -- and, crucially, whatever IS returned is a genuine root (residual
// ~ 0), never a phantom. Phase-2 (compensated Horner) should recover the rest;
// when it does, this assertion flips and the test should be updated.
TEST(SolverExtreme, WideDynamicRangeIsAKnownGap) {
    std::vector<cd> truth = { cd(1e-4, 0), cd(1.0, 0), cd(1e4, 0) };
    Polynomial P     = from_roots(truth);
    auto       found = find_roots(P, 1e-9);

    EXPECT_LT(total_multiplicity(found), 3)
        << "If this now finds all 3, the evaluator improved -- update the test.";
    EXPECT_GE(total_multiplicity(found), 1);

    // The roots it does keep are real (small residual) and finite -- the gap is
    // lost roots, never spurious ones.
    EXPECT_LT(max_relative_residual(P, found), 1e-6);
    for (const auto& f : found) {
        EXPECT_TRUE(std::isfinite(f.z.real()));
        EXPECT_TRUE(std::isfinite(f.z.imag()));
    }
}

// Coefficient overflow boundary. Roots at radius ~1e160 make Horner partials
// ~1e320+, overflowing double. The contract under test is robustness, not
// accuracy: the solver must TERMINATE and return finite output, never hang or
// emit NaN/Inf root coordinates.
TEST(SolverExtreme, OverflowBoundaryTerminatesCleanly) {
    std::vector<cd> truth = { cd(1e160, 0), cd(2e160, 0), cd(3e160, 0) };
    Polynomial      P     = from_roots(truth);

    auto t0    = std::chrono::steady_clock::now();
    auto found = find_roots(P, 1.0e150); // coarse eps matched to the scale
    EXPECT_LT(seconds_since(t0), 20.0) << "did not terminate promptly";

    for (const auto& f : found) {
        EXPECT_TRUE(std::isfinite(f.z.real())) << "NaN/Inf root coordinate";
        EXPECT_TRUE(std::isfinite(f.z.imag())) << "NaN/Inf root coordinate";
    }
    EXPECT_LE(total_multiplicity(found), 3) << "spurious extra roots";
}

// ===========================================================================
// Bucket D -- metamorphic invariants (hold for ANY polynomial)
// ===========================================================================

// Scaling all coefficients by a nonzero constant leaves the roots unchanged:
// c*P and P must yield the same root set.
TEST(SolverInvariant, CoefficientScaleInvariance) {
    std::vector<cd> truth = { cd(1, 1), cd(-2, 0.5), cd(0.3, -1.4), cd(3, 0) };
    Polynomial P = from_roots(truth);

    Polynomial Q = P;
    for (cd& c : Q.coeff) c *= cd(1e6, -3e5); // arbitrary nonzero complex scale

    auto fp = find_roots(P, 1e-9);
    auto fq = find_roots(Q, 1e-9);
    ASSERT_EQ(total_multiplicity(fp), 4);
    ASSERT_EQ(total_multiplicity(fq), 4);
    EXPECT_LT(max_error(truth, fp), 1e-6);
    EXPECT_LT(max_error(truth, fq), 1e-6);
}

// Real coefficients => roots come in conjugate pairs. Build a real polynomial
// from conjugate pairs and assert the recovered set is conjugate-symmetric.
TEST(SolverInvariant, ConjugateSymmetryForRealPolynomial) {
    std::vector<cd> truth = {
        cd(0.5, 1.3), cd(0.5, -1.3), cd(-2.0, 0.7), cd(-2.0, -0.7), cd(3.0, 0.0)
    };
    Polynomial P = from_roots(truth);
    // Constructed-real, but scrub any imaginary round-off so the input is
    // exactly real as the invariant assumes.
    for (cd& c : P.coeff) c = cd(c.real(), 0.0);

    auto found = find_roots(P, 1e-9);
    ASSERT_EQ(total_multiplicity(found), 5);

    // Every found root's conjugate must also be a found root.
    for (const auto& f : found) {
        cd     conj = std::conj(f.z);
        double best = 1e300;
        for (const auto& g : found) best = std::min(best, std::abs(conj - g.z));
        EXPECT_LT(best, 1e-6)
            << "no conjugate partner for " << f.z.real() << "+" << f.z.imag()
            << "i";
    }
}
