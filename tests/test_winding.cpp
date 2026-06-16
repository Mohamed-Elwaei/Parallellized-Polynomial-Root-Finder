// Tests for winding_count / count_roots.
#include "polyroots/polynomial.hpp"
#include "polyroots/winding.hpp"
#include <gtest/gtest.h>

using namespace polyroots;

TEST(Winding, CountsAllThreeRealRoots) {
    // (z-1)(z-2)(z-3): a box around [0,4] x [-1,1] holds all three.
    Polynomial p = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    Square box{ cd(2, 0), 2.5, 0, true };
    EXPECT_EQ(count_roots(p, box), 3);
    EXPECT_TRUE(box.trusted);
}

TEST(Winding, CountsZeroWhenBoxIsEmpty) {
    Polynomial p = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    // Box far from any root.
    Square box{ cd(10, 10), 1.0, 0, true };
    EXPECT_EQ(count_roots(p, box), 0);
}

TEST(Winding, IsolatesSingleRoot) {
    Polynomial p = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    Square box{ cd(2, 0), 0.4, 0, true }; // tight around z=2 only
    EXPECT_EQ(count_roots(p, box), 1);
}

TEST(Winding, CountsMultiplicityOfRepeatedRoot) {
    // (z-1)^3: a box around z=1 should report a winding number of 3.
    Polynomial p = from_roots({ cd(1, 0), cd(1, 0), cd(1, 0) });
    Square box{ cd(1, 0), 0.5, 0, true };
    EXPECT_EQ(count_roots(p, box), 3);
}

TEST(Winding, CleanContourHasSmallResidual) {
    Polynomial p = from_roots({ cd(1, 0), cd(2, 0), cd(3, 0) });
    Square box{ cd(2, 0), 2.5, 0, true };
    double residual = 1.0;
    bool   trusted  = false;
    winding_count(p, box, 4 * p.degree(), residual, trusted);
    EXPECT_LT(residual, 0.1);
    EXPECT_TRUE(trusted);
}

TEST(Winding, ComplexConjugatePairCounted) {
    // z^2 + 1 has roots +-i; a box around the origin captures both.
    Polynomial p(std::vector<cd>{ cd(1, 0), cd(0, 0), cd(1, 0) });
    Square box{ cd(0, 0), 1.5, 0, true };
    EXPECT_EQ(count_roots(p, box), 2);
}
