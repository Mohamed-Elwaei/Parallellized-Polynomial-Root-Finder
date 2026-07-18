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

## Status

- **Confirmed on hardware:** `p0`, `p2`.
- **Host-validated, not yet run on hardware:** `p1`, `scatter`, `pluggable_solver`.
- **Deferred (same as the CPU Phase-2 gaps):** cluster / high-multiplicity roots
  are dropped rather than resolved; fixed `sps` can miscount a root that *grazes*
  a sub-grid line (the CPU handles this with adaptive sampling); timing in
  `pluggable_solver` is coarse host wall-clock (switch to CUDA events for a
  precise winding-only comparison).
