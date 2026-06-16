// Tests for Polynomial: evaluation, derivative, construction, partial tracking.
#include "polyroots/polynomial.hpp"
#include <gtest/gtest.h>

using namespace polyroots;

namespace {
constexpr double TOL = 1e-12;
}

TEST(Polynomial, DegreeReportsCorrectly) {
    Polynomial p(std::vector<cd>{ cd(1, 0), cd(0, 0), cd(2, 0) }); // 2z^2 + 1
    EXPECT_EQ(p.degree(), 2);
}

TEST(Polynomial, HornerEvaluationMatchesHand) {
    // P(z) = 3 + 2z + z^2; P(2) = 3 + 4 + 4 = 11.
    Polynomial p(std::vector<cd>{ cd(3, 0), cd(2, 0), cd(1, 0) });
    cd v = p.eval(cd(2, 0));
    EXPECT_NEAR(v.real(), 11.0, TOL);
    EXPECT_NEAR(v.imag(), 0.0, TOL);
}

TEST(Polynomial, HornerHandlesComplexArgument) {
    // P(z) = z^2 + 1; P(i) = 0.
    Polynomial p(std::vector<cd>{ cd(1, 0), cd(0, 0), cd(1, 0) });
    cd v = p.eval(cd(0, 1));
    EXPECT_NEAR(std::abs(v), 0.0, TOL);
}

TEST(Polynomial, MaxPartialIsAtLeastFinalMagnitude) {
    Polynomial p(std::vector<cd>{ cd(3, 0), cd(2, 0), cd(1, 0) });
    double mp = 0.0;
    cd v = p.eval(cd(2, 0), mp);
    EXPECT_GE(mp, std::abs(v) - TOL);
}

TEST(Polynomial, DerivativeOfQuadratic) {
    // P(z) = z^2 + 2z + 3  ->  P'(z) = 2z + 2.
    Polynomial p(std::vector<cd>{ cd(3, 0), cd(2, 0), cd(1, 0) });
    Polynomial d = p.derivative();
    ASSERT_EQ(d.degree(), 1);
    EXPECT_NEAR(d.coeff[0].real(), 2.0, TOL); // constant term
    EXPECT_NEAR(d.coeff[1].real(), 2.0, TOL); // z coefficient
}

TEST(Polynomial, DerivativeOfConstantIsZero) {
    Polynomial p(std::vector<cd>{ cd(5, 0) });
    Polynomial d = p.derivative();
    EXPECT_NEAR(std::abs(d.coeff[0]), 0.0, TOL);
}

TEST(Polynomial, FromRootsReconstructsValues) {
    // (z-1)(z-2)(z-3); should vanish at each root.
    Polynomial p = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    ASSERT_EQ(p.degree(), 3);
    EXPECT_NEAR(std::abs(p.eval(cd(1, 0))), 0.0, TOL);
    EXPECT_NEAR(std::abs(p.eval(cd(2, 0))), 0.0, TOL);
    EXPECT_NEAR(std::abs(p.eval(cd(3, 0))), 0.0, TOL);
    // leading coefficient is monic
    EXPECT_NEAR(p.coeff.back().real(), 1.0, TOL);
}
