// ---------------------------------------------------------------------------
// solve_cli -- CPU solver as a benchmark CLI, sharing the I/O contract with the
// CUDA `solve` tool and the Python driver.
//
//   ./solve_cli <degree>     then read degree+1 "re im" pairs on stdin,
//                            DESCENDING (leading coefficient first).
//
// Output contract (machine-parseable; other lines are ignored by the driver):
//   T <milliseconds>
//   R <re> <im> <mult>       one per located root
// ---------------------------------------------------------------------------
#include "polyroots/polynomial.hpp"
#include "polyroots/solver.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace polyroots;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: solve_cli <degree>  "
                             "(then degree+1 're im' pairs on stdin, descending)\n");
        return 1;
    }
    int degree = std::atoi(argv[1]);
    if (degree < 1) { std::fprintf(stderr, "error: degree must be >= 1\n"); return 1; }

    std::vector<cd> desc;
    for (int i = 0; i <= degree; ++i) {
        double re, im;
        if (!(std::cin >> re >> im)) {
            std::fprintf(stderr, "error: expected %d coefficient pairs, got %d\n", degree + 1, i);
            return 1;
        }
        desc.push_back(cd(re, im));
    }

    Polynomial P;
    P.coeff.resize(degree + 1);
    for (int k = 0; k <= degree; ++k) P.coeff[k] = desc[degree - k];   // descending -> ascending
    if (std::abs(P.coeff.back()) == 0.0) {
        std::fprintf(stderr, "error: leading coefficient is zero\n");
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto roots = find_roots(P, 1e-10);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

    std::printf("T %.6f\n", ms);
    for (const auto& r : roots)
        std::printf("R %.17g %.17g %d\n", r.z.real(), r.z.imag(), r.mult);
    return 0;
}
