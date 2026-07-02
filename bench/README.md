# Benchmark harness

Races five implementations on a battery of polynomials, for **speed** and
**accuracy**:

- **cpu** — the CPU argument-principle solver (`build/solve_cli`)
- **gpu-naive / gpu-gather / gpu-scatter** — the three GPU winding policies (`solve`)
- **numpy** — `numpy.roots` (LAPACK companion-matrix eigenvalues; the practical baseline)
- **mpmath** — `mpmath.polyroots` (arbitrary precision; the accuracy oracle)

## I/O contract

All binaries read coefficients on stdin as `re im` pairs, **descending** (leading
coefficient first), and print:

```
T <milliseconds>
R <re> <im> <mult>      # one per located root
```

`np.poly(roots)` and `numpy.roots` use the same descending convention, so the
driver, NumPy, and mpmath all agree on coefficient order.

## Metrics

- **found** — number of roots returned (expanded by multiplicity); should equal the degree.
- **max_err** — worst error under a greedy nearest-neighbour match to the true roots
  (`LOST` = a true root went unmatched, i.e. a root was dropped).
- **residual** — `max |P(root)| / max|coeff|` (independent of knowing the truth).
- **time_ms** — wall-clock solve time.

## Running

### Locally (CPU + NumPy + mpmath; no GPU)
```bash
cmake --build build -j           # builds solve_cli
pip install numpy mpmath
python3 bench/benchmark.py        # auto-detects methods; skips the GPU if absent
```

### On Kaggle (all five)
Build `solve` from the CUDA source (see `cuda/README.md`), put `benchmark.py`
and the compiled `solve` / `solve_cli` in the working directory, then:
```python
!python3 benchmark.py --cpu ./solve_cli --gpu ./solve
```
Methods whose binaries aren't found are skipped, so the same script runs
anywhere.

## Notes / fairness

- The three GPU policies compute identical counts, so they return **identical
  roots** — only their *speed* differs; don't read into accuracy differences.
- GPU has fixed launch/transfer overhead: expect it to lose at low degree and
  only pay off at high degree (and in batch).
- `numpy`/`mpmath` use a different *algorithm* (eigenvalue), so this compares
  approaches, not just implementations.
- A double-precision reference (NumPy) shares this solver's conditioning limits
  on ill-conditioned cases (Wilkinson, tight clusters); **mpmath** is the
  independent truth there.
