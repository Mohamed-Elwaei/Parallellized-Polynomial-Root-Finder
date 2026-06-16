// End-to-end tests for find_roots, mirroring the validation battery.
#include "polyroots/polynomial.hpp"
#include "polyroots/solver.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

using namespace polyroots;
using polyroots::test::max_error;
using polyroots::test::total_multiplicity;

TEST(Solver, DistinctRealRoots) {
    std::vector<cd> truth = { cd(1, 0), cd(2, 0), cd(3, 0) };
    auto found = find_roots(from_roots(truth), 1e-8);
    EXPECT_EQ(total_multiplicity(found), 3);
    EXPECT_LT(max_error(truth, found), 1e-8);
}

TEST(Solver, RootsOfUnityDegree5) {
    int n = 5;
    Polynomial P;
    P.coeff.assign(n + 1, cd(0, 0));
    P.coeff[0] = cd(-1, 0);
    P.coeff[n] = cd(1, 0);
    std::vector<cd> truth;
    for (int k = 0; k < n; ++k)
        truth.push_back(std::polar(1.0, TWO_PI * k / n));
    auto found = find_roots(P, 1e-8);
    EXPECT_EQ(total_multiplicity(found), n);
    EXPECT_LT(max_error(truth, found), 1e-7);
}

TEST(Solver, RootsOfUnityDegree30) {
    int n = 30;
    Polynomial P;
    P.coeff.assign(n + 1, cd(0, 0));
    P.coeff[0] = cd(-1, 0);
    P.coeff[n] = cd(1, 0);
    std::vector<cd> truth;
    for (int k = 0; k < n; ++k)
        truth.push_back(std::polar(1.0, TWO_PI * k / n));
    auto found = find_roots(P, 1e-8);
    EXPECT_EQ(total_multiplicity(found), n);
    EXPECT_LT(max_error(truth, found), 1e-6);
}

TEST(Solver, RepeatedRootsDetectMultiplicity) {
    // (z-1)^3 (z+2)^2.
    std::vector<cd> truth = { cd(1, 0), cd(1, 0), cd(1, 0),
                              cd(-2, 0), cd(-2, 0) };
    auto found = find_roots(from_roots(truth), 1e-8);
    EXPECT_EQ(total_multiplicity(found), 5);

    // We should recover one box at z=1 (mult 3) and one at z=-2 (mult 2).
    int multAtOne = 0, multAtNegTwo = 0;
    for (const auto& f : found) {
        if (std::abs(f.z - cd(1, 0)) < 1e-3)  multAtOne   += f.mult;
        if (std::abs(f.z - cd(-2, 0)) < 1e-3) multAtNegTwo += f.mult;
    }
    EXPECT_EQ(multAtOne, 3);
    EXPECT_EQ(multAtNegTwo, 2);
}

TEST(Solver, ClusteredRootsResolved) {
    std::vector<cd> truth = {
        cd(0.0, 1.0), cd(0.0, -1.0), cd(2.0, 0.0),
        cd(0.5, 0.5), cd(0.5001, 0.5), cd(-3.0, 2.0)
    };
    auto found = find_roots(from_roots(truth), 1e-10);
    EXPECT_EQ(total_multiplicity(found), 6);
    EXPECT_LT(max_error(truth, found), 1e-6);
}

TEST(Solver, WilkinsonIsTheDocumentedHardCase) {
    // Wilkinson is the canonical ill-conditioned case and the motivation for
    // the planned Phase-2 robustness work (certified disk criterion + scaled/
    // compensated Horner). With the current Phase-1 evaluator, catastrophic
    // cancellation corrupts the winding integral for the larger roots: their
    // boxes register as empty and get pruned, so the solver silently LOSES most
    // of the 20 roots, recovering only the few smallest, best-conditioned ones.
    // This test pins that behaviour so a future fix shows up as a change here.
    std::vector<cd> truth;
    for (int k = 1; k <= 20; ++k) truth.push_back(cd(k, 0));
    auto found = find_roots(from_roots(truth), 1e-8);

    // Phase 1 recovers only a handful of roots, well short of 20.
    EXPECT_LT(total_multiplicity(found), 20)
        << "If this now reaches 20, the evaluator improved -- update the test.";
    EXPECT_GE(total_multiplicity(found), 1);

    // The roots it does recover are the small ones, and they are accurate.
    for (const auto& f : found) {
        double nearestInt = std::round(f.z.real());
        EXPECT_NEAR(f.z.real(), nearestInt, 1e-6);
        EXPECT_NEAR(f.z.imag(), 0.0, 1e-6);
        EXPECT_LE(nearestInt, 10.0); // only the lower half survives
    }
}
