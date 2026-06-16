// Tests for the root-enclosing bounds.
#include "polyroots/polynomial.hpp"
#include "polyroots/root_bound.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace polyroots;

// A bound is valid iff every root lies within it (with a small slack).
static void expect_bound_encloses(const std::vector<cd>& roots, double bound) {
    for (cd r : roots)
        EXPECT_LE(std::abs(r), bound * (1.0 + 1e-9))
            << "root " << r.real() << "+" << r.imag() << "i exceeds bound "
            << bound;
}

TEST(RootBound, CauchyEnclosesSimpleRoots) {
    std::vector<cd> r = { cd(1, 0), cd(2, 0), cd(3, 0) };
    expect_bound_encloses(r, cauchy_bound(from_roots(r)));
}

TEST(RootBound, FujiwaraEnclosesSimpleRoots) {
    std::vector<cd> r = { cd(1, 0), cd(2, 0), cd(3, 0) };
    expect_bound_encloses(r, fujiwara_bound(from_roots(r)));
}

TEST(RootBound, BothEncloseComplexRoots) {
    std::vector<cd> r = { cd(0, 1), cd(0, -1), cd(-3, 2), cd(2, 0) };
    Polynomial p = from_roots(r);
    expect_bound_encloses(r, cauchy_bound(p));
    expect_bound_encloses(r, fujiwara_bound(p));
}

TEST(RootBound, FujiwaraMuchTighterOnWilkinson) {
    std::vector<cd> r;
    for (int k = 1; k <= 20; ++k) r.push_back(cd(k, 0));
    Polynomial p = from_roots(r);
    // Cauchy blows up enormously on Wilkinson (~1e6+); Fujiwara is far tighter
    // (~420 here), keeping the initial contour in a numerically safe range.
    EXPECT_GT(cauchy_bound(p), 1e6);
    EXPECT_LT(fujiwara_bound(p), 1000.0);
    EXPECT_LT(fujiwara_bound(p), cauchy_bound(p) / 1000.0);
    // and Fujiwara still encloses every root.
    expect_bound_encloses(r, fujiwara_bound(p));
}

TEST(RootBound, RootBoundTakesTheMinimum) {
    std::vector<cd> r;
    for (int k = 1; k <= 20; ++k) r.push_back(cd(k, 0));
    Polynomial p = from_roots(r);
    EXPECT_DOUBLE_EQ(root_bound(p),
                     std::min(cauchy_bound(p), fujiwara_bound(p)));
}
