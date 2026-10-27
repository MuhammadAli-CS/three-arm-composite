"""How non-injective is each outer map, measured rather than assumed.

The proposition predicts that a bijective g gives the composite model no
information advantage.  Testing that needs the independent variable to be
injectivity itself, not a proxy like polynomial degree -- degree also controls
how wild the function is, which affects how well any GP fits it.

So for each map this counts, over the distribution of y that the problem
actually produces, how many values of h could have produced it:

    nbar  = E[ |g^{-1}(y)| ]   averaged over y ~ g(h(X)), X ~ uniform

restricted to the range h actually visits.  nbar = 1 means injective where it
matters; nbar = 2 means a typical observation leaves two candidates.

    python three_arm/injectivity.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.quasirandom import SobolEngine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.test_functions import DTYPE, GROUPS, make_problem   # noqa: E402


def fiber_sizes(problem, n_probe: int = 4000, grid: int = 20001):
    """|g^{-1}(y)| for each observed y, counted by sign changes.

    Counting crossings of the scalar map on a dense grid works for any g, not
    just polynomials, and needs no root solver.  The grid spans the range h
    actually reaches, padded slightly, so preimages outside the support are
    not counted -- a branch the prior never visits is not a real ambiguity.
    """
    if problem.outer.k != 1:
        # |g^{-1}(y)| for g: R^k -> R is a (k-1)-dimensional set, not a count,
        # so the crossing counter does not apply.  It used to crash here with
        # an IndexError; say so instead.
        raise ValueError(f"{problem.name} has k={problem.outer.k}; the fiber "
                         "counter is only defined for a single intermediate")
    X = SobolEngine(problem.d, scramble=True, seed=404).draw(n_probe).to(DTYPE)
    h = problem.h(X)[:, 0]
    lo, hi = float(h.min()), float(h.max())
    pad = 0.02 * (hi - lo)
    z = torch.linspace(lo - pad, hi + pad, grid, dtype=DTYPE)
    gz = problem.outer.fn(z.unsqueeze(-1)).numpy()          # (grid,)
    ys = problem.outer.fn(h.unsqueeze(-1)).numpy()          # (n_probe,)

    # For each y, count sign changes of g(z) - y along the grid.
    # A flat region is not a crossing, so counting sign changes silently misses
    # it -- which is how -max(h,0) came out as injective when half the domain
    # maps to one value.  Flatness has to be judged by the SLOPE, though, not
    # by |g(z) - y| against a global tolerance: h^5 is merely very small near
    # z = 0, not constant, and an absolute threshold called 11% of mono5's
    # domain flat when mono5 is strictly increasing.
    #
    # So mark the cells where g is genuinely constant -- zero finite difference
    # relative to the total variation of g -- and require them to cover a real
    # fraction of the domain before calling a level set infinite.  That
    # separates relu (half the domain exactly constant) from mono5 (a single
    # stationary point, measure zero).
    dg = np.abs(np.diff(gz))
    span = max(float(np.abs(gz).max() - np.abs(gz).min()), 1e-300)
    flat_cell = dg < 1e-12 * span
    flat_frac = float(flat_cell.mean())
    counts = np.empty(len(ys))
    tol = 1e-12 * max(float(np.abs(gz).max()), 1e-300)
    for i, y in enumerate(ys):
        d = gz - y
        if flat_frac > 0.01 and np.any(flat_cell & (np.abs(d[:-1]) < tol)):
            counts[i] = np.inf
            continue
        counts[i] = int(np.count_nonzero(np.signbit(d[:-1]) != np.signbit(d[1:])))
    return counts, (lo, hi)


def summarize(name, d=6, draw_seed=0):
    p = make_problem(name, d=d, lengthscale=0.5, draw_seed=draw_seed)
    c, (lo, hi) = fiber_sizes(p)
    finite = c[np.isfinite(c)]
    return dict(map=name, kind=p.outer.kind,
                nbar=float(finite.mean()) if len(finite) else float("inf"),
                frac_infinite=float((~np.isfinite(c)).mean()),
                frac_multi=float((c > 1).mean()),
                nmax=int(finite.max()) if len(finite) else -1,
                hrange=(lo, hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=GROUPS["bijectivity"])
    ap.add_argument("--draw-seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)
    print(f"{'map':9s} {'kind':11s} {'nbar':>6s} {'% y ambiguous':>14s} {'max':>4s}  h range")
    print("-" * 64)
    for m in a.maps:
        r = summarize(m, draw_seed=a.draw_seed)
        print(f"{r['map']:9s} {r['kind']:11s} {r['nbar']:6.3f} {r['frac_multi']:13.1%} "
              f"{r['nmax']:4d}  [{r['hrange'][0]:+.2f}, {r['hrange'][1]:+.2f}]")
    print("\nnbar = average number of h values consistent with one observation.\n"
          "1.000 means injective over the region h actually reaches.")


if __name__ == "__main__":
    main()
