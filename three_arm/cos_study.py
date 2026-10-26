"""Why cos(4h) breaks the pattern, and whether it is a budget problem.

cos_w4 has the most preimages of any map (nbar = 9.75) and nearly the
smallest composite gain.  The proposed explanation is that Composite's edge
depends on its uncertainty about h being small compared with the scale over
which g varies.  For cos(omega h) that scale is a half-period, pi/omega, so
the quantity that matters is

    omega * sd(h | data)   versus   pi

At omega = 4 and 45 observations this is about 2.6 radians -- most of a
half-period -- so knowing h tells you almost nothing about cos(4h), and the
predictive on f collapses to the marginal.

That is a testable prediction rather than a story: sd(h | data) shrinks as
data arrives, so every map should cross back below the threshold given enough
of it, and the gain should appear.  This sweeps n to find out.

    python three_arm/cos_study.py
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
from three_arm.test_functions import DTYPE, make_problem            # noqa: E402

warnings.filterwarnings("ignore")
N_SAMPLES = 2048


def _gp(X, y):
    m = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    except Exception:
        m.eval()
    return m


def one(problem, omega, n_train, n_test, seed):
    X = SobolEngine(problem.d, scramble=True, seed=seed).draw(n_train + n_test).to(DTYPE)
    Xtr, Xte = X[:n_train], X[n_train:]
    htr, hte = problem.h(Xtr)[:, 0], problem.h(Xte)[:, 0]
    ftr, fte = problem.f(Xtr), problem.f(Xte)

    gp_f = _gp(Xtr, ftr)
    with torch.no_grad():
        d_rmse = float(((gp_f.posterior(Xte).mean.squeeze(-1) - fte) ** 2).mean().sqrt())

    gp_h = _gp(Xtr, htr)
    with torch.no_grad():
        post = gp_h.posterior(Xte)
    mu = post.mean.squeeze(-1)
    sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    gen = torch.Generator().manual_seed(seed * 13 + 5)
    eps = torch.randn(N_SAMPLES, n_test, generator=gen, dtype=DTYPE)
    draws = problem.g((mu + sd * eps).unsqueeze(-1))
    c_rmse = float(((draws.mean(0) - fte) ** 2).mean().sqrt())

    # how informative is Composite's predictive on f, relative to f's own spread
    ratio = float(draws.std(0).mean() / fte.std())
    return float(sd.mean()), omega * float(sd.mean()), d_rmse, c_rmse, ratio, float(fte.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+",
                    default=["cos_w1", "cos_w2", "cos_w4"])
    ap.add_argument("--ns", type=int, nargs="+", default=[20, 40, 80, 160, 320, 640])
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    OM = {"cos_w1": 1.0, "cos_w2": 2.0, "cos_w4": 4.0}
    print("Composite's edge should appear once omega*sd(h) drops below about pi.\n")
    print(f"{'map':8s} {'n':>5s} {'sd(h|data)':>11s} {'omega*sd':>9s} "
          f"{'predictive ratio':>17s} {'RMSE direct':>12s} {'RMSE comp':>10s} {'gain %':>8s}")
    for m in a.maps:
        om = OM.get(m, 0.0)
        for n in a.ns:
            rows = []
            for d_ in range(a.draws):
                p = make_problem(m, d=6, lengthscale=0.5, draw_seed=d_)
                for r in range(a.reps):
                    rows.append(one(p, om, n, a.n_test, 1000 * d_ + r))
            z = [st.mean([r[k] for r in rows]) for k in range(6)]
            gain = 100.0 * (z[2] - z[3]) / z[2]
            flag = "  <-- below pi" if z[1] < math.pi else ""
            print(f"{m:8s} {n:5d} {z[0]:11.3f} {z[1]:9.2f} {z[4]:17.3f} "
                  f"{z[2]:12.4f} {z[3]:10.4f} {gain:+8.1f}{flag}", flush=True)
        print()
    print("predictive ratio = sd of Composite's predictive on f, over f's own sd.")
    print("1.0 means knowing h has told it nothing about the objective.")


if __name__ == "__main__":
    main()
