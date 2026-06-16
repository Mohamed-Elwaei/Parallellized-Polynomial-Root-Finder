// Tests for modified Newton polishing and its convergence behaviour.
#include "polyroots/newton.hpp"
#include "polyroots/polynomial.hpp"
#include <gtest/gtest.h>

using namespace polyroots;

TEST(Newton, ConvergesOnSimpleRoot) {
    // (z-1)(z-2)(z-3); polish a guess near z=2.
    Polynomial p  = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    Polynomial dp = p.derivative();
    cd z = newton_polish(p, dp, cd(2.1, 0.05), /*mult=*/1);
    EXPECT_NEAR(std::abs(z - cd(2, 0)), 0.0, 1e-12);
}

TEST(Newton, ConvergesOnDoubleRootWithMultTwo) {
    // (z-1)^2; modified Newton with mult=2 restores quadratic convergence.
    Polynomial p  = from_roots({ cd(1, 0), cd(1, 0) });
    Polynomial dp = p.derivative();
    cd z = newton_polish(p, dp, cd(1.2, 0.1), /*mult=*/2);
    EXPECT_NEAR(std::abs(z - cd(1, 0)), 0.0, 1e-10);
}

TEST(Newton, ModifiedStepBeatsPlainAtMultipleRoot) {
    // With the same iteration budget, mult=3 should reach the triple root of
    // (z-1)^3 far more accurately than plain (mult=1) Newton.
    Polynomial p  = from_roots({ cd(1, 0), cd(1, 0), cd(1, 0) });
    Polynomial dp = p.derivative();
    cd start = cd(1.3, 0.0);
    cd zPlain    = newton_polish(p, dp, start, /*mult=*/1, /*iters=*/8);
    cd zModified = newton_polish(p, dp, start, /*mult=*/3, /*iters=*/8);
    double errPlain    = std::abs(zPlain - cd(1, 0));
    double errModified = std::abs(zModified - cd(1, 0));
    EXPECT_LT(errModified, errPlain);
    EXPECT_NEAR(errModified, 0.0, 1e-9);
}

TEST(Newton, ConvergesOnComplexRoot) {
    // z^2 + 1, root at i.
    Polynomial p(std::vector<cd>{ cd(1, 0), cd(0, 0), cd(1, 0) });
    Polynomial dp = p.derivative();
    cd z = newton_polish(p, dp, cd(0.05, 1.1), /*mult=*/1);
    EXPECT_NEAR(std::abs(z - cd(0, 1)), 0.0, 1e-12);
}
