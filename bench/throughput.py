#!/usr/bin/env python3
"""
Throughput benchmark: solve a BATCH of N polynomials and compare
polynomials/second for the GPU batched solver vs a NumPy serial loop.

This is the framing where the GPU is meant to win: NumPy processes the batch
one polynomial at a time; the GPU floods every SM with all N at once.

Usage:
    python3 throughput.py -N 10000 -d 10 --gpu ./batched
The GPU binary is skipped if not present (so the NumPy side runs anywhere).
"""
import argparse, subprocess, time, os
import numpy as np

def gen_batch(N, deg, seed=0, scale=1.0):
    """N random degree-`deg` polynomials, built from random complex roots.

    Roots are uniform on the square [-1,1]^2, then multiplied by `scale`.
    Scaling does NOT change the difficulty in any mathematical sense -- relative
    geometry, and hence conditioning, is identical. It is a knob for testing the
    solver's SCALE INVARIANCE: every threshold compared against a length must
    track the root bound, and a fixed scale hides it when they don't (that is
    exactly how the isoThresh bug survived so long).
    """
    rng = np.random.default_rng(seed)
    return [np.poly(scale * (rng.uniform(-1, 1, deg) + 1j * rng.uniform(-1, 1, deg)))
            for _ in range(N)]

def float_root_band(deg):
    """(min, max) root magnitude --float can represent at this degree.

    Coefficients are cast to float for the winding, and |c_0| ~ s^deg, so the
    usable band is FLT_MIN^(1/deg) .. FLT_MAX^(1/deg). It narrows from BOTH
    ends as degree rises. Verified against hardware at scales 1e-6..1e6: this
    predicts pass/fail exactly, where a |P|-on-the-contour bound did not
    (intermediate overflow far from the roots turns out to be benign).
    """
    return (1.1754944e-38 ** (1.0 / deg), 3.4028235e38 ** (1.0 / deg))

def run_numpy(polys):
    t0 = time.perf_counter()
    for c in polys:
        np.roots(c)                    # one polynomial at a time (serial)
    return time.perf_counter() - t0

def run_gpu(binary, polys, deg, extra=None):
    N = len(polys)
    lines = []
    for c in polys:                    # descending coeffs, "re im" per line
        for k in range(deg + 1):
            lines.append(f"{float(c[k].real):.17g} {float(c[k].imag):.17g}")
    stdin = "\n".join(lines) + "\n"
    r = subprocess.run([binary, str(N), str(deg)] + (extra or []), input=stdin,
                       capture_output=True, text=True, timeout=600)
    keys = ("SOLVE_MS", "THROUGHPUT", "MAXRESIDUAL", "ROOTS",
            "SETUP_MS", "WINDING_MS", "TRIAGE_MS", "NEWTON_MS")
    out = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if p and p[0] in keys:
            out[p[0]] = float(p[1])
    return out, r.stderr

def scale_sweep(gpu, N, deg, scales):
    """Solve the SAME polynomials at wildly different scales.

    Relative geometry is identical at every scale, so a scale-invariant solver
    must return the same completeness throughout. Also shows the --float
    representability wall: past `float_root_limit(deg)` the cast to single
    precision overflows and the binary warns.
    """
    lo, hi = float_root_band(deg)
    print(f"\nscale sweep: N={N} degree={deg}")
    print(f"  roots are uniform on [-1,1]^2 * scale, so max |root| ~ 1.41*scale")
    print(f"  --float band at degree {deg}: |root| in [{lo:.3g}, {hi:.3g}]\n")
    print(f"  {'scale':>9}  {'double':>18}  {'float':>18}  note")
    for s in scales:
        polys = gen_batch(N, deg, scale=s)
        cells, warned, fcomp = [], "", None
        for prec in ("--double", "--float"):
            g, err = run_gpu(gpu, polys, deg, [prec, "--policy", "gather"])
            if "ROOTS" not in g:
                cells.append("failed"); continue
            exp = N * deg
            cells.append(f"{int(g['ROOTS'])}/{exp} ({100*g['ROOTS']/exp:6.2f}%)")
            if prec == "--float":
                fcomp = g["ROOTS"] / exp
                if "OVERFLOW" in err:  warned = "overflow"
                if "UNDERFLOW" in err: warned = "underflow"
        # Cross-check the guard against what actually happened. Both directions
        # matter: a miss ships garbage, a false alarm rejects valid work.
        broke = fcomp is not None and fcomp < 0.90
        note = f"guard: {warned}" if warned else ""
        if broke and not warned:  note = "!! BROKE with NO warning (guard missed it)"
        elif warned and not broke: note += "  !! false alarm (float was fine)"
        print(f"  {s:>9.0e}  {cells[0]:>18}  {cells[1]:>18}  {note}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="./batched")
    ap.add_argument("-N", type=int, default=10000)
    ap.add_argument("-d", "--degree", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all roots by this (tests scale invariance)")
    ap.add_argument("--scale-sweep", action="store_true",
                    help="sweep several scales instead of racing the policies")
    a = ap.parse_args()

    if a.scale_sweep:
        if not os.path.exists(a.gpu):
            print(f"gpu binary '{a.gpu}' not found"); return
        scale_sweep(a.gpu, a.N, a.degree, [1e-6, 1e-3, 1e0, 1e3, 1e6])
        return

    print(f"batch: N={a.N} polynomials, degree={a.degree}, root scale={a.scale:g}")
    polys = gen_batch(a.N, a.degree, scale=a.scale)

    tnp = run_numpy(polys)
    np_tput = a.N / tnp
    print(f"  numpy (serial loop):  {tnp*1e3:9.1f} ms   {np_tput:12.0f} polys/sec")

    if not os.path.exists(a.gpu):
        print(f"  gpu   : '{a.gpu}' not found (build + run on Kaggle)")
        return
    # Race: double/naive (validated reference), then float x {naive, gather,
    # scatter} to isolate the winding-policy speedup at the winning precision.
    # (The arg method is fixed to atan2 — measurement showed it doesn't matter.)
    combos = [("--double", "atan2", "naive")] + [("--float", "atan2", p)
                                                 for p in ("naive", "gather", "scatter")]
    for prec, arg, pol in combos:
        g, err = run_gpu(a.gpu, polys, a.degree, [prec, "--arg", arg, "--policy", pol])
        tag = f"{prec.lstrip('-')}/{pol}"
        if "THROUGHPUT" not in g:
            print(f"  gpu ({tag}): failed ->", err[:300]); continue
        print(f"  gpu ({tag:<15}):  {g['SOLVE_MS']:9.1f} ms   {g['THROUGHPUT']:12.0f} polys/sec"
              f"   (max residual {g['MAXRESIDUAL']:.1e})  {g['THROUGHPUT']/np_tput:.1f}x vs numpy")
        if "ROOTS" in g:
            exp = a.N * a.degree
            miss = exp - int(g["ROOTS"])
            note = "" if miss == 0 else f"   <-- INCOMPLETE ({miss} lost)"
            # 4 decimals: at N=10000 a 0.03% loss must not round to a clean 100.0%
            print(f"      roots found: {int(g['ROOTS'])}/{exp} ({100*g['ROOTS']/exp:.4f}%){note}")
            if g["ROOTS"] < exp and err.strip():                  # surface the binary's stderr
                print(f"      stderr: {err.strip()[:300]}")
        if "WINDING_MS" in g:
            print(f"      breakdown: setup {g['SETUP_MS']:.1f} | winding {g['WINDING_MS']:.1f} | "
                  f"triage {g['TRIAGE_MS']:.1f} | newton {g['NEWTON_MS']:.1f}  (ms)")

if __name__ == "__main__":
    main()
