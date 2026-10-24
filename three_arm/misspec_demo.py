"""Why Direct loses on a cubic, in one regression experiment.

No optimisation, no acquisition: just fit and predict.  100 points, 80 for
training and 20 held out, repeated over random splits and function draws.

    Direct     a GP fitted to f, scored on f
    Composite  a GP fitted to h, scored on h (inner) and, after pushing the
               posterior through the known g, on f (outer)

The point is the pair of numbers for Composite.  If its inner RMSE is tiny --
h is a GP sample, so a GP models it almost exactly -- while its outer RMSE
still beats Direct's, then Direct's disadvantage cannot be a shortage of data
or a worse fit.  It is that f = g(h) is not a Gaussian process when g is
nonlinear, so the model Direct is fitting has the wrong shape.

z^3 is the cleanest case: strictly increasing, so it is bijective and
observing h conveys no information Direct lacks.  Anything Composite gains is
model shape alone.

    python three_arm/misspec_demo.py
    python three_arm/misspec_demo.py --maps mono3 root3 shiftsq_c0
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.arms import KnownPrior                              # noqa: E402
from three_arm.test_functions import DTYPE, make_problem           # noqa: E402

warnings.filterwarnings("ignore")
N_SAMPLES = 4096


def _gp(X, y):
    m = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    except Exception:
        m.eval()
    return m


def rmse(a, b):
    return float(((a - b) ** 2).mean().sqrt())


def one_split(problem, n_train, n_test, seed):
    X = SobolEngine(problem.d, scramble=True, seed=seed).draw(n_train + n_test).to(DTYPE)
    Xtr, Xte = X[:n_train], X[n_train:]
    Htr, Hte = problem.h(Xtr), problem.h(Xte)
    ftr, fte = problem.f(Xtr), problem.f(Xte)

    # Direct: one GP on the objective itself
    gp_f = _gp(Xtr, ftr)
    with torch.no_grad():
        direct_outer = rmse(gp_f.posterior(Xte).mean.squeeze(-1), fte)

    # Composite: one GP per intermediate, then push through the known g
    gen = torch.Generator().manual_seed(seed * 17 + 1)
    means, sds = [], []
    inner = []
    for j in range(Htr.shape[-1]):
        gp_h = _gp(Xtr, Htr[:, j])
        with torch.no_grad():
            post = gp_h.posterior(Xte)
        mu = post.mean.squeeze(-1)
        means.append(mu)
        sds.append(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
        inner.append(rmse(mu, Hte[:, j]))
    eps = torch.randn(N_SAMPLES, len(means), n_test, generator=gen, dtype=DTYPE)
    draws = torch.stack([m_ + s_ * eps[:, k] for k, (m_, s_) in enumerate(zip(means, sds))],
                        dim=-1)                                   # (S, n_test, k)
    comp_outer = rmse(problem.g(draws).mean(0), fte)

    # The same, but handed the generator's true kernel instead of fitting one
    prior = KnownPrior.from_problem(problem)
    L = prior.unit_chol(Xtr)
    mu_o, sd_o = prior.conditional(Xtr, Htr.unsqueeze(0), Xte, L)
    oracle_inner = rmse(mu_o[0, :, 0], Hte[:, 0])
    draws_o = mu_o + sd_o * eps[:, 0, :].unsqueeze(-1) if mu_o.shape[-1] == 1 else None
    if draws_o is None:
        eps2 = torch.randn(N_SAMPLES, n_test, mu_o.shape[-1], generator=gen, dtype=DTYPE)
        draws_o = mu_o + sd_o * eps2
    oracle_outer = rmse(problem.g(draws_o).mean(0), fte)

    # Normalise each RMSE by the spread of what it is predicting, so the
    # inner and outer numbers are on the same footing: h has sd 1 while f is
    # percentile-scaled to a much smaller spread, and the raw figures are not
    # otherwise comparable.
    sd_f = float(fte.std()); sd_h = float(Hte.std())
    return (direct_outer / sd_f, st.mean(inner) / sd_h, comp_outer / sd_f,
            oracle_inner / sd_h, oracle_outer / sd_f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=["identity", "mono3", "root3"])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--splits", type=int, default=15)
    ap.add_argument("--draws", type=int, default=4)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    print(f"{a.n_train} training points, {a.n_test} held out, "
          f"{a.draws} draws x {a.splits} splits = {a.draws * a.splits} fits per map\n")
    print(f"{'map':10s} {'DIRECT':>14s} | {'COMPOSITE (fitted)':>28s} | "
          f"{'COMPOSITE (true kernel)':>28s}")
    print(f"{'':10s} {'rmse outer':>14s} | {'rmse inner':>13s} {'rmse outer':>14s} | "
          f"{'rmse inner':>13s} {'rmse outer':>14s}")
    print("-" * 94)
    for m in a.maps:
        D, I, O, OI, OO = [], [], [], [], []
        for draw in range(a.draws):
            p = make_problem(m, d=6, lengthscale=0.5, draw_seed=draw)
            for s in range(a.splits):
                d_, i_, o_, oi_, oo_ = one_split(p, a.n_train, a.n_test, 1000 * draw + s)
                D.append(d_); I.append(i_); O.append(o_); OI.append(oi_); OO.append(oo_)
        f = lambda v: f"{st.mean(v):.4f}+/-{st.stdev(v) / math.sqrt(len(v)):.4f}"
        print(f"{m:10s} {f(D):>14s} | {f(I):>13s} {f(O):>14s} | {f(OI):>13s} {f(OO):>14s}")
    print("\nAll RMSE on the 20 held-out points, lower is better.")
    print("Composite's inner RMSE being tiny while its outer RMSE still beats")
    print("Direct's is the signature of misspecification: the data is not the")
    print("problem, the shape of the model is.")


if __name__ == "__main__":
    main()
