# polyroots — CUDA port (prototypes)

GPU implementation of the argument-principle root finder, developed against the
CPU reference in the parent directory (which stays the correctness oracle).

These are **standalone prototypes** built to be pasted into a **Kaggle GPU
notebook** — CUDA doesn't run on macOS, so the workflow is: edit here, run on
Kaggle. Each `.cu` is self-checking (validates against known roots and prints
`PASS`/`FAIL`).

## Shared math

`polyroots.cuh` holds the `__host__ __device__` primitives used by every
prototype **and** callable from plain host code, so the CPU and GPU run the same
leaf math: `poly_eval` (Horner), `perim_point`, `winding_count`, `edge_phase`,
`newton_polish`, `unwrap`, plus `cmplx` (= `thrust::complex<double>`), `Cell`,
`kPI`, and the `CK` error-check macro.

## The progression

| File | What it is | Parallelization |
|------|------------|-----------------|
| `p0_grid_winding.cu`     | winding count of every cell in one grid | 1 thread / cell (recompute edges) |
| `p1_edge_winding.cu`     | same counts, edge-shared (**gather**)    | 1 thread / edge → H/V tables, no atomics |
| `scatter_winding.cu`     | same counts, edge-shared (**scatter**)   | 1 thread / edge → `atomicAdd` into M |
| `p2_subdivide_solve.cu`  | **full solver**: BFS subdivide + Newton  | per-level winding kernel + polish kernel |
| `solve.cu`               | **CLI tool: one BFS, three swappable policies** | `winding_kernel<Policy>` (template), 1 block / square, K×K sub-grid |

`p0`/`p1`/`scatter` are three interchangeable ways to answer *"how many roots in
each cell?"*. `p2` is the pipeline that repeatedly asks that and polishes.
`solve.cu` is the culmination: one BFS driver, the winding policy chosen at
compile time via a template (so it stays inlined — no function-pointer overhead),
with the Cauchy/Fujiwara **root bound wired in**, driven from the command line so
it solves an **arbitrary** polynomial.

### `solve.cu` usage

```
./solve <-naive|-gather|-scatter|-race> <degree>   # then degree+1 "re im" pairs on stdin
./solve -selfcheck                                  # built-in known polynomial
```
Coefficients are entered **descending** (leading first); a degree-`n` polynomial
needs `n+1` complex pairs. Example — `z^5 - 1`:
```
printf "1 0\n0 0\n0 0\n0 0\n0 0\n-1 0\n" | ./solve -race 5
```
`-race` runs all three policies on the same input; each root is printed with its
residual `|P(z)|`, so you can sanity-check without knowing the true roots.

## Running on Kaggle

1. New Notebook → **Settings → Accelerator → GPU T4**.
2. Write the shared header once:
   ```python
   %%writefile polyroots.cuh
   # paste cuda/polyroots.cuh
   ```
3. Write and run a prototype (example: the `solve` CLI tool):
   ```python
   %%writefile solve.cu
   # paste the file
   ```
   ```python
   !nvcc -O2 -arch=sm_75 solve.cu -o solve
   !./solve -selfcheck                                   # built-in known polynomial
   !printf "1 0\n0 0\n0 0\n0 0\n0 0\n-1 0\n" | ./solve -race 5   # z^5 - 1
   !compute-sanitizer ./solve -selfcheck                 # catches memory/race bugs
   ```
`sm_75` is the T4's architecture (use `sm_60` for a P100, or drop `-arch`).

The batched throughput solver is built the same way (it needs `-std=c++17`
for the compile-time `--arg` dispatch):
```python
!nvcc -O2 -std=c++17 -arch=sm_75 batched_solve.cu -o batched
# 10k degree-10 polynomials, float winding, gather policy:
!python3 bench/throughput.py -N 10000 -d 10 --gpu ./batched
# or one config directly:  ./batched 10000 10 --float --arg atan2 --policy gather
```
`--arg` selects the winding's argument method — `atan2` (default, validated
reference), `approx` (cheap polynomial atan2), or `quadrant` (sign-based, no
transcendental). Measurement showed the arg method makes **no** difference on
GPU (the winding is Horner-bound, not transcendental-bound).

`--policy` selects how the sub-grid edges are evaluated — `naive` (default; each
subcell samples its own 4 edges, interior edges recomputed twice), `gather`
(fill a shared canonical edge table once, each subcell reads its 4 edges), or
`scatter` (each edge computed once, atomic-added into its two neighbour cells).
`gather`/`scatter` do ~1.7x fewer Horner evals than `naive`. `throughput.py`
races the three policies at float precision.

### Measured (T4, N=10000 polynomials, degree 10)

> ⚠️ **These timings predate the `isoThresh` tightening** (see below) and were
> taken at ~96–99% completeness. The *relative* ordering of the policies still
> holds, but the absolute throughput is now lower (roughly 2.4x the cells) and
> completeness is 100%. Re-measure before quoting these numbers.

| config | winding ms | polys/sec | vs numpy |
|-----------------|-----------:|----------:|---------:|
| double / naive  |     1362.7 |     7 155 |     1.0x |
| float / naive   |      208.7 |    40 341 |     5.4x |
| float / scatter |      186.4 |    44 697 |     6.0x |
| **float / gather** | **162.2** | **50 068** | **6.7x** |

**Completeness comes first.** `isoThresh` is now `R/(K*deg)` — one subdivision
level tighter than `R/deg`. Because subdivision divides by `K = 8` per level,
`R/deg` and the old absolute `0.1` selected the *same* too-coarse level, and
both handed Newton cells that were still too large to seed reliably. That —
not clustering — was the long-standing few-percent root loss. Tightening it
takes a 300-polynomial host batch from ~96% to **100.00%**, at ~2.4x the cells.

Three findings worth keeping:

1. **`--float` is the big win (~5x).** fp64 runs at 1/32 rate on a T4, and the
   winding only has to *count* roots — Newton still polishes in double, so the
   final accuracy is unchanged (max residual 7.2e-15 in every config).
2. **`--arg` makes no measurable difference.** Replacing `atan2` with a cheap
   polynomial, or removing the transcendental entirely (`quadrant`), leaves the
   time unchanged: the winding is bound by the **Horner evaluation**, not the
   argument. All three return identical roots. Kept as a selectable knob because
   it definitively answers the question, not because it pays.
3. **`gather` beats `scatter`.** Both evaluate the same 144 edges once, but
   scatter's shared-memory `atomicAdd`s contend (4 edges hit each subcell
   accumulator). `gather` is also *marginally more complete* (98919 vs naive's
   98916 roots) because a shared edge is computed once, so both neighbouring
   subcells see bit-identical boundary values instead of two independent
   float recomputations.

**Recommended: `--float --policy gather`.** Note `scatter` is nondeterministic
run-to-run (float atomic ordering); `naive`/`gather` are reproducible.

## Host regression checks (no GPU needed)

The kernels need a GPU, but every numerical decision they make is ordinary
double arithmetic. [`tests/host_checks.cpp`](tests/host_checks.cpp) mirrors that
leaf math on the host so it runs anywhere:

```bash
g++ -O2 -std=c++17 cuda/tests/host_checks.cpp -o host_checks && ./host_checks
```

It pins three properties, each of which was expensive to discover:

1. **Scale invariance** — the same roots scaled over `1e-9 .. 1e+9` must all
   give 10/10. Four *absolute* constants each broke this independently
   (`isoThresh`, `minHalf`, the Newton step tolerance, the dedup tolerance);
   with them restored the check fails 1/10 at scale `1e-9`, 8/10 at `1e-3`, and
   **passes at scale >= 1** — which is precisely why a benchmark fixed at
   roots in [-1,1]^2 never caught it.
2. **Argument methods agree** — `atan2` / `approx` / `quadrant` must find the
   same roots.
3. **Edge decomposition** — `gather` and `scatter` must reproduce `naive`'s
   counts on all 64 subcells. This is where an edge orientation sign error
   would surface; on hardware it would just silently miscount.

Exits non-zero on failure, so it can be wired into CI.

## Status

- **Confirmed on hardware:** `p0`, `p2`, `batched_solve` (all 3 precisions x
  3 arg methods x 3 policies; see the measured table above).
- **Host-validated, not yet run on hardware:** `p1`, `scatter`, `pluggable_solver`.
- **Root loss (was ~1-4%) is fixed** — it was `isoThresh` being one subdivision
  level too coarse, *not* the cluster limitation it was long assumed to be. A
  300-polynomial host batch now finds 100.00% of roots; `host_checks` check 4
  pins this so it cannot silently regress.
- **Deferred (same as the CPU Phase-2 gaps):** cluster / high-multiplicity roots
  are dropped rather than resolved; fixed `sps` can miscount a root that *grazes*
  a sub-grid line (the CPU handles this with adaptive sampling); timing in
  `pluggable_solver` is coarse host wall-clock (switch to CUDA events for a
  precise winding-only comparison).
- **Next unexploited redundancy:** `gather` removed the shared-*edge* recompute,
  but the (K+1)^2 grid *corners* are still evaluated once per incident edge (up
  to 4x). Precomputing them per block would shave more, with diminishing returns.
