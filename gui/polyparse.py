"""
polyparse -- the "compiler frontend": parse a raw polynomial expression and
lower it to coefficient representation (the IR the solver consumes).

Coefficients are COMPLEX (ascending: c[k] is the coefficient of x^k). The AST is
evaluated with polynomial arithmetic (add / convolve / power).

Functions (sin, cos, ln, sqrt, abs, ...) and constants (pi, e, i) ARE allowed --
but a FUNCTION'S ARGUMENT MUST BE A CONSTANT (no x), because sin(constant) is a
number (a valid coefficient) while sin(x) is not a polynomial. So:
    sin(1+2i) x^2      OK   (coefficient = sin(1+2i))
    4 i cos(0) x^3 + i OK   (= 4i x^3 + i)
    tan(x) x^2         ERROR (argument contains x)
Exponents must be non-negative integer constants (x^i, x^-2, x^2.5 -> error).
"""
from __future__ import annotations
import cmath, math

# --------------------------------------------------------------------------- #
# Constants and functions (all complex-valued, applied to a constant argument).
# --------------------------------------------------------------------------- #
CONST_VAL = {"pi": math.pi, "e": math.e}
FUNC_IMPL = {
    "sin": cmath.sin, "cos": cmath.cos, "tan": cmath.tan,
    "cot": lambda z: 1 / cmath.tan(z), "sec": lambda z: 1 / cmath.cos(z),
    "csc": lambda z: 1 / cmath.sin(z),
    "asin": cmath.asin, "acos": cmath.acos, "atan": cmath.atan,
    "sinh": cmath.sinh, "cosh": cmath.cosh, "tanh": cmath.tanh,
    "ln": cmath.log, "log": lambda z: cmath.log10(z), "exp": cmath.exp,
    "sqrt": cmath.sqrt, "abs": lambda z: complex(abs(z), 0),
}
FUNCS, CONSTS = set(FUNC_IMPL), set(CONST_VAL)
_NAMES = sorted(FUNCS | CONSTS, key=len, reverse=True)   # longest match first

# --------------------------------------------------------------------------- #
# Polynomial arithmetic on ascending COMPLEX coefficient lists.
# --------------------------------------------------------------------------- #
def p_add(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0j) + (b[i] if i < len(b) else 0j) for i in range(n)]

def p_neg(a): return [-x for x in a]

def p_mul(a, b):
    r = [0j] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            r[i + j] += ai * bj
    return r

def p_pow(a, n):
    r = [1 + 0j]
    for _ in range(n):
        r = p_mul(r, a)
    return r

def _trim(c):
    while len(c) > 1 and abs(c[-1]) == 0.0:
        c.pop()
    return c

_ONLY_POLY = ("This calculator only handles polynomials in x — use numbers, x, i, "
              "the constants pi and e, functions of a CONSTANT (e.g. sin(1+2i)), "
              "and + - * ^ ( ).")

# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #
def tokenize(s: str):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif s[i:i+2] == "**":
            toks.append(("^", "^")); i += 2
        elif c in "+-*^()|":
            toks.append((c, c)); i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            toks.append(("num", float(s[i:j]))); i = j
        elif c.isalpha():
            name = next((nm for nm in _NAMES if s.startswith(nm, i)), None)
            if name in FUNCS:   toks.append(("func", name));  i += len(name)
            elif name in CONSTS: toks.append(("const", name)); i += len(name)
            elif c in "xX":     toks.append(("x", "x")); i += 1
            elif c in "iIjJ":   toks.append(("i", "i")); i += 1
            else: raise ValueError(_ONLY_POLY)
        else:
            raise ValueError(_ONLY_POLY)
    toks.append(("end", None))
    return toks

# --------------------------------------------------------------------------- #
# Recursive-descent parser.
# --------------------------------------------------------------------------- #
class Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i]
    def adv(self):  tok = self.t[self.i]; self.i += 1; return tok

    def parse(self):
        c = self.expr()
        if self.peek()[0] != "end":
            raise ValueError(f"unexpected {self.peek()[1]!r}")
        return _trim(c)

    def expr(self):
        sign = 1
        if self.peek()[0] == "+": self.adv()
        elif self.peek()[0] == "-": self.adv(); sign = -1
        c = self.term()
        if sign < 0: c = p_neg(c)
        while self.peek()[0] in ("+", "-"):
            op = self.adv()[0]
            t = self.term()
            c = p_add(c, t) if op == "+" else p_add(c, p_neg(t))
        return c

    def term(self):
        # implicit multiplication by juxtaposition. '|' is NOT here: it is not
        # directional, so a *closing* |...| must not be read as a new factor.
        c = self.power()
        while self.peek()[0] in ("*", "num", "x", "i", "const", "func", "("):
            if self.peek()[0] == "*": self.adv()
            c = p_mul(c, self.power())
        return c

    def power(self):
        b = self.atom()
        if self.peek()[0] == "^":
            self.adv()
            b = p_pow(b, _as_exponent(self.power()))
        return b

    def atom(self):
        k, v = self.peek()
        if k == "num":   self.adv(); return [complex(v, 0)]
        if k == "x":     self.adv(); return [0j, 1 + 0j]
        if k == "i":     self.adv(); return [1j]
        if k == "const": self.adv(); return [complex(CONST_VAL[v], 0)]
        if k == "-":     self.adv(); return p_neg(self.atom())
        if k == "func":
            self.adv()
            if self.peek()[0] != "(":
                raise ValueError(f"expected '(' after {v}")
            self.adv(); arg = self.expr()
            if self.peek()[0] != ")":
                raise ValueError("missing ')'")
            self.adv()
            return [FUNC_IMPL[v](_as_constant(arg, v))]
        if k == "|":                                   # |...|  absolute value
            self.adv(); arg = self.expr()
            if self.peek()[0] != "|":
                raise ValueError("missing closing '|'")
            self.adv()
            return [complex(abs(_as_constant(arg, "|·|")), 0)]
        if k == "(":
            self.adv(); c = self.expr()
            if self.peek()[0] != ")":
                raise ValueError("missing ')'")
            self.adv(); return c
        raise ValueError(_ONLY_POLY)

def _as_constant(coeffs, fname):
    """A function/abs argument must be a constant (degree 0), else it has x in it."""
    if len(coeffs) != 1:
        raise ValueError(f"the argument of {fname} must be a constant (it can't "
                         f"contain x) — otherwise the result isn't a polynomial")
    return coeffs[0]

def _as_exponent(coeffs):
    if len(coeffs) != 1:
        raise ValueError("the exponent must be a constant, not something with x")
    e = coeffs[0]
    if abs(e.imag) > 1e-12 or e.real < 0 or e.real != int(e.real):
        val = e.real if abs(e.imag) < 1e-12 else e
        raise ValueError(f"the exponent must be a non-negative integer for this to "
                         f"be a polynomial (got {val})")
    return int(e.real)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def normalize_mathfield(s: str) -> str:
    """MathLive/LaTeX output -> the infix form parse_polynomial wants."""
    reps = [
        ("\\operatorname", ""),
        ("\\arcsin", "asin"), ("\\arccos", "acos"), ("\\arctan", "atan"),
        ("\\sinh", "sinh"), ("\\cosh", "cosh"), ("\\tanh", "tanh"),
        ("\\sin", "sin"), ("\\cos", "cos"), ("\\tan", "tan"),
        ("\\cot", "cot"), ("\\sec", "sec"), ("\\csc", "csc"),
        ("\\ln", "ln"), ("\\log", "log"), ("\\exp", "exp"), ("\\sqrt", "sqrt"),
        ("\\pi", "pi"), ("\\exponentialE", "e"), ("\\mathrm{e}", "e"),
        ("\\imaginaryI", "i"), ("\\mathi", "i"),
        ("\\cdot", "*"), ("\\times", "*"),
        ("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", ""), ("\\ ", ""),
        ("{", "("), ("}", ")"), ("**", "^"),
    ]
    for a, b in reps:
        s = s.replace(a, b)
    return s

def parse_polynomial(text: str):
    """text -> ascending COMPLEX coefficient list [c0, c1, ..., cn]."""
    if not text.strip():
        raise ValueError("empty input")
    return Parser(tokenize(text)).parse()

def cauchy_bound(c):
    n = len(c) - 1; cn = abs(c[n])
    return 1.0 + max((abs(c[k]) / cn for k in range(n)), default=0.0)

def fujiwara_bound(c):
    n = len(c) - 1; cn = abs(c[n]); m = 0.0
    for k in range(1, n + 1):
        r = abs(c[n - k]) / cn
        if k == n: r *= 0.5
        m = max(m, r ** (1.0 / k))
    return 2.0 * m

def root_bound(c):
    return 1.0 if len(c) < 2 else min(cauchy_bound(c), fujiwara_bound(c))

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ok = ["(x+5)^6", "sin(1+2i)x^2", "4i cos(0) x^3 + i", "pi x^2 + e",
          "|1+2i| x", "sqrt(4) x - 1", "x^(1+1)"]
    bad = ["tan(x) x^2", "x^i", "x^-2", "x^2.5", "ln(x)"]
    for t in ok:
        c = parse_polynomial(t)
        print(f"OK   {t:20s} -> deg {len(c)-1}, {[complex(round(v.real,4),round(v.imag,4)) for v in c]}")
    for t in bad:
        try:
            parse_polynomial(t); print(f"!!!  {t:20s} -> parsed but should have errored")
        except ValueError as e:
            print(f"ERR  {t:20s} -> {str(e)[:55]}")
