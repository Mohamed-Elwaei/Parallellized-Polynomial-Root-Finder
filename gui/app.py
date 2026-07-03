"""
Polynomial root calculator -- GUI.

Frontend (polyparse):  raw expression -> coefficient IR
Backend (solve_cli):   coefficients   -> roots
Output:                text + a plot of the complex plane with the roots,
                       darker dots for higher multiplicity, bordered by the
                       root-enclosing contour.

Run:  python3 gui/app.py
"""
from __future__ import annotations
import os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polyparse import parse_polynomial, root_bound

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVE_CLI = os.environ.get("SOLVE_CLI", os.path.join(REPO, "build", "solve_cli"))  # CPU
SOLVE_GPU = os.environ.get("SOLVE_GPU", os.path.join(REPO, "solve"))              # CUDA build

# backend name -> (binary, extra args).  All share the same stdin/T/R contract.
BACKENDS = {
    "cpu":     (SOLVE_CLI, []),           # the CPU argument-principle solver
    "naive":   (SOLVE_GPU, ["-naive"]),   # GPU: per-cell winding
    "gather":  (SOLVE_GPU, ["-gather"]),  # GPU: edge-shared (tables)
    "scatter": (SOLVE_GPU, ["-scatter"]), # GPU: edge-shared (atomics)
}

def fmt_coeff(c):
    """Format a complex coefficient compactly (real if the imag part is ~0)."""
    if abs(c.imag) < 1e-12:
        return f"{c.real:+g}"
    return f"+({c.real:g}{c.imag:+g}i)"

# --------------------------------------------------------------------------- #
# Backend call: coefficients (ascending, complex) -> [(complex root, mult)].
# --------------------------------------------------------------------------- #
def solve(coeffs, backend="cpu"):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (use one of {list(BACKENDS)})")
    binary, flags = BACKENDS[backend]
    degree = len(coeffs) - 1
    if degree < 1:
        raise ValueError("that's a constant, not a polynomial with roots")
    if not os.path.exists(binary):
        hint = ("build it: cmake --build build" if backend == "cpu"
                else "GPU binary needs a CUDA machine: nvcc cuda/solve.cu -o solve")
        raise FileNotFoundError(f"{backend} solver not found at {binary} ({hint})")
    desc = coeffs[::-1]                                   # descending for the CLI
    stdin = "".join(f"{v.real:.17g} {v.imag:.17g}\n" for v in desc)  # complex coefficients
    r = subprocess.run([binary, *flags, str(degree)], input=stdin,
                       capture_output=True, text=True, timeout=120)
    roots = []
    for line in r.stdout.splitlines():
        if line.startswith("R "):
            p = line.split()
            roots.append((complex(float(p[1]), float(p[2])), int(p[3])))
    if not roots and r.stderr:
        raise RuntimeError(r.stderr.strip())
    return roots

# --------------------------------------------------------------------------- #
# Plot: complex plane, contour box, roots (darker = higher multiplicity).
# --------------------------------------------------------------------------- #
def make_figure(coeffs, roots, fig=None):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    B = root_bound(coeffs)
    lim = B * 1.15
    if fig is None:
        fig = Figure(figsize=(5.5, 5.5))
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.grid(True, color="#eeeeee", zorder=0)

    # root-enclosing contour (the square the solver bounds roots within)
    ax.add_patch(Rectangle((-B, -B), 2 * B, 2 * B, fill=False,
                           edgecolor="#2a6f97", ls="--", lw=1.4, zorder=2,
                           label="root bound"))

    maxm = max((m for _, m in roots), default=1)
    for z, m in roots:
        # darker + larger for higher multiplicity
        shade = 0.75 - 0.55 * ((m - 1) / max(maxm - 1, 1))    # 0.75 (light) -> 0.20 (dark)
        color = (shade * 0.4, shade * 0.3, shade)             # bluish, darkens with m
        ax.scatter(z.real, z.imag, s=70 + 45 * (m - 1), color=color,
                   edgecolors="white", linewidths=0.8, zorder=4)
        if m > 1:
            ax.annotate(f"x{m}", (z.real, z.imag), textcoords="offset points",
                        xytext=(7, 6), fontsize=9, color="#333333", zorder=5)

    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title(f"{len(roots)} root(s) of a degree-{len(coeffs)-1} polynomial")
    return fig

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def main():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Polynomial Root Calculator")

    top = ttk.Frame(root, padding=8); top.pack(fill="x")
    ttk.Label(top, text="Polynomial:").pack(side="left")
    entry = ttk.Entry(top, width=40); entry.pack(side="left", padx=6)
    entry.insert(0, "(x^2 + 2x + 5)^4")
    backend_var = tk.StringVar(value="cpu")
    ttk.Label(top, text="Backend:").pack(side="left")
    ttk.Combobox(top, textvariable=backend_var, values=list(BACKENDS),
                 width=8, state="readonly").pack(side="left", padx=4)

    body = ttk.Frame(root, padding=(8, 0)); body.pack(fill="both", expand=True)
    txt = tk.Text(body, width=34, height=24, font=("Menlo", 11)); txt.pack(side="left", fill="y")

    fig = Figure(figsize=(5.5, 5.5))
    canvas = FigureCanvasTkAgg(fig, master=body)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def run(*_):
        txt.delete("1.0", "end")
        expr = entry.get()
        try:
            coeffs = parse_polynomial(expr)
            roots = solve(coeffs, backend_var.get())
        except Exception as e:
            txt.insert("end", f"Error:\n{e}\n"); return
        txt.insert("end", f"Parsed (degree {len(coeffs)-1}):\n")
        for k in range(len(coeffs) - 1, -1, -1):
            if coeffs[k] != 0:
                txt.insert("end", f"  {fmt_coeff(coeffs[k])} x^{k}\n")
        txt.insert("end", f"\nRoots ({len(roots)}):\n")
        for z, m in sorted(roots, key=lambda r: (r[0].real, r[0].imag)):
            mult = f"  (x{m})" if m > 1 else ""
            txt.insert("end", f"  {z.real:+.6f} {z.imag:+.6f}i{mult}\n")
        make_figure(coeffs, roots, fig=fig)
        canvas.draw()

    ttk.Button(top, text="Solve", command=run).pack(side="left")
    entry.bind("<Return>", run)
    run()
    root.mainloop()

if __name__ == "__main__":
    main()
