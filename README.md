# polyroots — argument-principle subdivision root finder (Phase 1, CPU)

A single-threaded C++17 reference implementation of a complex-analytic
polynomial root finder. It locates all roots of a polynomial by recursively
subdividing a bounding square and using the **argument principle** (winding
numbers) to count the roots in each sub-region, then polishing the isolated
roots with a multiplicity-aware **modified Newton** step.

This is the correctness oracle and CPU baseline for the CUDA port now underway
in [`cuda/`](cuda/). The control flow is deliberately written as a breadth-first
work-list so it maps directly onto a GPU design: each `while`-loop iteration
corresponds to one kernel launch over the whole active queue.

> 📄 **Full write-up** — the method, the GPU parallelization, benchmarks, and
> limitations are explained with figures in
> [`paper/paper.pdf`](paper/paper.pdf) (source: [`paper/paper.tex`](paper/paper.tex)).

## How it works

The finder never solves for roots algebraically; it *counts* and then *locates*
them by exploiting a fact from complex analysis:

1. **Bound.** Cauchy/Fujiwara bounds give a square guaranteed to contain every
   root of the polynomial.
2. **Count by winding.** Cauchy's **argument principle** says that as `z` walks
   once counter-clockwise around the boundary of a region, the image `P(z)`
   winds around the origin exactly once per enclosed root. Sampling the boundary
   and summing the change in `arg P` gives that integer count directly — no root
   values needed.
3. **Subdivide.** Split each square into four. A sub-square with count `0` is
   discarded, `1` is a resolved isolation, `≥2` is subdivided again. The counts
   always sum back to the polynomial's degree, which is the search's invariant.
4. **Polish.** Once a root is isolated in a small cell, a multiplicity-aware
   **modified Newton** step refines it to full precision.

Because each sub-square's count depends only on *its own* boundary, the regions
are embarrassingly parallel — which is what the [`cuda/`](cuda/) port exploits,
with three winding strategies (per-cell, edge-gather, atomic-scatter). See the
[paper](paper/paper.pdf) for the full derivation, diagrams, and the numerical
edge cases where this breaks down.

## Layout

```
include/polyroots/   public headers (one per module)
  constants.hpp        shared constants (pi, eps_machine)
  polynomial.hpp       Polynomial: Horner eval, derivative, from_roots
  root_bound.hpp       Cauchy / Fujiwara root-enclosing bounds
  winding.hpp          Square, winding_count, count_roots
  newton.hpp           modified Newton polishing
  solver.hpp           find_roots (the top-level subdivision driver)
src/                 implementations, one .cpp per header
apps/demo.cpp        human-facing smoke test over a battery of polynomials
tests/               GoogleTest suites, one per module (incl. test_solver_stress)
cuda/                GPU port prototypes + shared header (see cuda/README.md)
bench/               speed/accuracy benchmark: CPU vs GPU vs NumPy vs mpmath
gui/                 web calculator: parse an expression -> solve -> plot roots
docs/                operating-envelope / limitations report
CMakeLists.txt       primary build (fetches GoogleTest automatically)
Makefile             lightweight lib + demo build
```

## Build & run

### CMake (recommended — fetches GoogleTest for you)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/demo                  # run the demo
ctest --test-dir build        # run all unit tests
```

### Makefile (quick, dependency-free for the demo)

```bash
make run                      # build + run the demo
make test                     # build + run tests (needs a system GoogleTest)
```

## Validation status

| Case                         | Result                                            |
|------------------------------|---------------------------------------------------|
| Distinct real roots          | machine precision                                 |
| Roots of unity (deg 5/12/30) | machine precision                                 |
| Repeated roots (z-1)^3(z+2)^2| correct multiplicity at the accuracy ceiling      |
| Clustered roots              | resolved to ~1e-13                                |
| Wilkinson (deg 20)           | **fails** — catastrophic cancellation (Phase-2)   |

The Wilkinson failure is expected and is the motivation for the planned Phase-2
robustness work (certified disk criterion + scaled/compensated Horner). The
`Solver.WilkinsonIsTheDocumentedHardCase` test pins the current behaviour so a
future improvement shows up as a deliberate change.

`tests/test_solver_stress.cpp` pushes the solver along each axis (degree, root
magnitude, dynamic range, clustering, multiplicity) until it breaks; the
measured operating envelope is written up in
[`docs/`](docs/polyroots_limitations_report.pdf). The solver also carries a hard
work-list cap so pathological inputs (e.g. very high multiplicity) terminate
rather than hang.

## CUDA port

[`cuda/`](cuda/) contains the GPU implementation, developed against this CPU
reference as the oracle. It progresses from a single winding kernel to a full
pluggable solver (one BFS, three swappable winding strategies), with the shared
leaf math written `__host__ __device__` so CPU and GPU stay in agreement. The
prototypes run in a Kaggle GPU notebook — see [`cuda/README.md`](cuda/README.md).

## GUI

[`gui/`](gui/) is a web calculator that ties the whole pipeline together: a
Desmos-style **MathLive** math field (vendored locally, so it works offline)
parses an expression to coefficients, feeds them to the solver, and shows the
roots as text and on the complex plane (dashed root-bound contour, dots darker
for higher multiplicity). It supports complex coefficients, functions of a
constant argument (e.g. `sin(1+2i)`), the constants `pi`/`e`, and a CPU/GPU
backend selector. Run it with `python3 gui/server.py` (needs Flask + matplotlib)
and open the printed URL.

## Testing framework

Tests use **GoogleTest** — the C++ analog of Python's `unittest`. CMake pulls
it in via `FetchContent`, so no manual install is needed. The suites are split
by module (`test_polynomial`, `test_root_bound`, `test_winding`,
`test_newton`, `test_solver`) and registered with CTest individually via
`gtest_discover_tests`, so you can filter and run them like any CTest project.
