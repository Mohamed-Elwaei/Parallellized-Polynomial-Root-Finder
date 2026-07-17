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

def gen_batch(N, deg, seed=0):
    """N random degree-`deg` polynomials (from random complex roots)."""
    rng = np.random.default_rng(seed)
    return [np.poly(rng.uniform(-1, 1, deg) + 1j * rng.uniform(-1, 1, deg)) for _ in range(N)]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="./batched")
    ap.add_argument("-N", type=int, default=10000)
    ap.add_argument("-d", "--degree", type=int, default=10)
    a = ap.parse_args()

    print(f"batch: N={a.N} polynomials, degree={a.degree}")
    polys = gen_batch(a.N, a.degree)

    tnp = run_numpy(polys)
    np_tput = a.N / tnp
    print(f"  numpy (serial loop):  {tnp*1e3:9.1f} ms   {np_tput:12.0f} polys/sec")

    if not os.path.exists(a.gpu):
        print(f"  gpu   : '{a.gpu}' not found (build + run on Kaggle)")
        return
    for flag in ("--double", "--float"):
        g, err = run_gpu(a.gpu, polys, a.degree, [flag])
        tag = flag.lstrip("-")
        if "THROUGHPUT" not in g:
            print(f"  gpu ({tag}): failed ->", err[:300]); continue
        print(f"  gpu ({tag:<6})     :  {g['SOLVE_MS']:9.1f} ms   {g['THROUGHPUT']:12.0f} polys/sec"
              f"   (max residual {g['MAXRESIDUAL']:.1e})  {g['THROUGHPUT']/np_tput:.1f}x vs numpy")
        if "ROOTS" in g:
            exp = a.N * a.degree
            flag = "" if g["ROOTS"] == exp else "   <-- INCOMPLETE (lost roots!)"
            print(f"      roots found: {int(g['ROOTS'])}/{exp} ({100*g['ROOTS']/exp:.1f}%){flag}")
            if g["ROOTS"] < exp and err.strip():                  # surface the binary's stderr
                print(f"      stderr: {err.strip()[:300]}")
        if "WINDING_MS" in g:
            print(f"      breakdown: setup {g['SETUP_MS']:.1f} | winding {g['WINDING_MS']:.1f} | "
                  f"triage {g['TRIAGE_MS']:.1f} | newton {g['NEWTON_MS']:.1f}  (ms)")

if __name__ == "__main__":
    main()
