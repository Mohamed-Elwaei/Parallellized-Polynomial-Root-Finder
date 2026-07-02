#!/usr/bin/env python3
"""
Benchmark harness: race the CPU solver, the three GPU winding policies, NumPy,
and mpmath (multiprecision oracle) on a battery of polynomials, for speed and
accuracy.

I/O contract with the C++/CUDA binaries: coefficients are fed on stdin as
"re im" pairs, DESCENDING (leading first); binaries print
    T <milliseconds>
    R <re> <im> <mult>
Other output lines are ignored.

Usage:
    python3 benchmark.py                 # auto-detects available methods
    python3 benchmark.py --cpu ./build/solve_cli --gpu ./solve
Methods that aren't found are skipped (so it runs locally with just CPU+NumPy).
"""
import argparse, subprocess, time, math, cmath, os, sys
import numpy as np

# --------------------------------------------------------------------------- #
# Method runners: each returns (roots, time_ms) where roots = [(complex, mult)].
# --------------------------------------------------------------------------- #
def _parse(out):
    roots, t = [], None
    for line in out.splitlines():
        if line.startswith("T "):
            t = float(line[2:])
        elif line.startswith("R "):
            p = line.split()
            roots.append((complex(float(p[1]), float(p[2])), int(p[3])))
    return roots, t

def run_binary(cmd, degree, desc):
    # float() + .17g so numpy scalars serialize as plain "1.0", not numpy-2.0's
    # repr "np.float64(1.0)" (which the C++/CUDA parsers can't read).
    stdin = "".join(f"{float(c.real):.17g} {float(c.imag):.17g}\n" for c in desc)
    try:
        r = subprocess.run(cmd + [str(degree)], input=stdin,
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, float("inf")
    return _parse(r.stdout)

def run_numpy(desc):
    a = np.array([complex(c) for c in desc])
    t0 = time.perf_counter(); r = np.roots(a); ms = (time.perf_counter() - t0) * 1e3
    return [(complex(z), 1) for z in r], ms

def run_mpmath(desc):
    try:
        import mpmath
    except ImportError:
        return None, None
    mpmath.mp.dps = 40
    coeffs = [mpmath.mpc(c.real, c.imag) for c in desc]
    t0 = time.perf_counter()
    try:
        r = mpmath.polyroots(coeffs, maxsteps=300, extraprec=300)
    except Exception:
        return None, (time.perf_counter() - t0) * 1e3
    ms = (time.perf_counter() - t0) * 1e3
    return [(complex(z), 1) for z in r], ms

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def expand(roots):
    pts = []
    for z, m in roots:
        pts += [z] * max(m, 1)
    return pts

def match_error(found_pts, truth_pts):
    """Greedy nearest-neighbour match; worst per-root error. Huge if undercount."""
    used = [False] * len(found_pts); maxerr = 0.0
    for t in truth_pts:
        best, bi = 1e300, -1
        for j, f in enumerate(found_pts):
            if not used[j]:
                d = abs(t - f)
                if d < best: best, bi = d, j
        if bi >= 0: used[bi] = True; maxerr = max(maxerr, best)
        else:       maxerr = 1e300
    return maxerr

def residual(desc, pts):
    a = np.array([complex(c) for c in desc])
    scale = max(np.abs(a).max(), 1e-300)
    return max((abs(np.polyval(a, z)) / scale for z in pts), default=0.0)

# --------------------------------------------------------------------------- #
# Polynomial families -> (name, degree, desc_coeffs, truth_points_or_None)
# --------------------------------------------------------------------------- #
def poly_from_roots(roots):
    return list(np.poly(roots))            # DESCENDING coefficients

def battery():
    tests = []
    rng = np.random.default_rng(0)
    # baseline: roots of unity
    for n in (5, 10, 20, 50):
        roots = [cmath.exp(2j * math.pi * k / n) for k in range(n)]
        c = [0.0] * (n + 1); c[0] = 1.0; c[-1] = -1.0
        tests.append((f"unity_deg{n}", n, c, roots))
    # random well-separated in a disk
    for n in (8, 20):
        roots = []
        while len(roots) < n:
            z = complex(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
            if abs(z) <= 1.5 and all(abs(z - w) > 0.3 for w in roots):
                roots.append(z)
        tests.append((f"random_deg{n}", n, poly_from_roots(roots), roots))
    # Wilkinson
    for n in (10, 20):
        roots = [float(k) for k in range(1, n + 1)]
        tests.append((f"wilkinson_deg{n}", n, poly_from_roots(roots), roots))
    # clustered pair
    for d in (1e-3, 1e-6):
        roots = [0.3, 0.3 + d, -1.0, 2.0 + 1j]
        tests.append((f"cluster_{d:.0e}", 4, poly_from_roots(roots), roots))
    # multiplicity
    for m in (2, 3, 5):
        roots = [2.0] * m + [-3.0]
        tests.append((f"mult{m}", m + 1, poly_from_roots(roots), roots))
    # edge cases
    tests.append(("linear", 1, [2.0, -6.0], [3.0]))                    # 2z-6
    tests.append(("origin", 3, poly_from_roots([0.0, 1.0, 2.0]), [0.0, 1.0, 2.0]))
    tests.append(("complex_coeffs", 3,
                  poly_from_roots([1 + 1j, -2 + 0.5j, 0.3 - 1.4j]), [1 + 1j, -2 + 0.5j, 0.3 - 1.4j]))
    tests.append(("wide_range", 3, poly_from_roots([1e-3, 1.0, 1e3]), [1e-3, 1.0, 1e3]))
    return tests

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default="./build/solve_cli")
    ap.add_argument("--gpu", default="./solve")        # CUDA `solve` (on Kaggle)
    args = ap.parse_args()

    methods = []                                        # (label, callable(desc, degree))
    if os.path.exists(args.cpu):
        methods.append(("cpu", lambda d, deg: run_binary([args.cpu], deg, d)))
    for pol in ("naive", "gather", "scatter"):
        if os.path.exists(args.gpu):
            methods.append((f"gpu-{pol}",
                            lambda d, deg, p=pol: run_binary([args.gpu, f"-{p}"], deg, d)))
    methods.append(("numpy",  lambda d, deg: run_numpy(d)))
    methods.append(("mpmath", lambda d, deg: run_mpmath(d)))

    print("method availability:", ", ".join(m[0] for m in methods))
    hdr = f"{'polynomial':<16}{'method':<10}{'found':>6}{'max_err':>11}{'residual':>11}{'time_ms':>10}"
    print(hdr); print("-" * len(hdr))

    for name, degree, desc, truth in battery():
        truth_pts = list(truth) if truth is not None else None
        for label, fn in methods:
            roots, ms = fn(desc, degree)
            if roots is None:
                print(f"{name:<16}{label:<10}{'--':>6}{'n/a':>11}{'n/a':>11}{'--':>10}")
                continue
            pts = expand(roots)
            err = match_error(pts, truth_pts) if truth_pts is not None else float("nan")
            res = residual(desc, pts)
            errs = f"{err:.2e}" if err == err and err < 1e299 else ("LOST" if err == err else "n/a")
            tstr = f"{ms:.3f}" if ms is not None and ms != float("inf") else "TIMEOUT"
            print(f"{name:<16}{label:<10}{len(pts):>6}{errs:>11}{res:>11.2e}{tstr:>10}")
        print()

if __name__ == "__main__":
    main()
