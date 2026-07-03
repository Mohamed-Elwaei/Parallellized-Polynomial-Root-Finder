"""
Web backend for the polynomial root calculator.

Serves a page with a MathLive math-input field (the Desmos-style entry), and a
/solve endpoint that: normalizes the MathLive expression -> parses to
coefficients (polyparse) -> runs the solver (solve_cli) -> returns the roots and
a rendered plot, all shown on the same page.

Run:  python3 gui/server.py     then open http://127.0.0.1:5000
"""
import os, sys, io, base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify, send_from_directory
from polyparse import parse_polynomial, normalize_mathfield
from app import solve, make_figure, fmt_coeff, BACKENDS   # reuse backend call + plotting

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")

@app.route("/vendor/<path:fname>")
def vendor(fname):                    # serve the local MathLive bundle + fonts
    return send_from_directory(os.path.join(HERE, "vendor"), fname)

@app.route("/solve", methods=["POST"])
def do_solve():
    data = request.json or {}
    expr = data.get("expr", "")
    backend = data.get("backend", "cpu")
    try:
        infix  = normalize_mathfield(expr)
        coeffs = parse_polynomial(infix)
        roots  = solve(coeffs, backend)
    except Exception as e:
        return jsonify(error=str(e))

    fig = make_figure(coeffs, roots)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plot_b64 = base64.b64encode(buf.getvalue()).decode()

    # human-readable polynomial (descending)
    terms = []
    for k in range(len(coeffs) - 1, -1, -1):
        if coeffs[k] != 0:
            terms.append(f"{fmt_coeff(coeffs[k])}x^{k}" if k else fmt_coeff(coeffs[k]))
    return jsonify(
        degree=len(coeffs) - 1,
        poly=" ".join(terms),
        roots=[[z.real, z.imag, m] for (z, m) in
               sorted(roots, key=lambda r: (r[0].real, r[0].imag))],
        plot=plot_b64,
    )

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")     # headless rendering for the server
    port = int(os.environ.get("PORT", 8000))   # 8000 avoids macOS's port-5000 (AirPlay)
    app.run(port=port, use_reloader=False)
