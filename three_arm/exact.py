"""Exact posteriors by enumerating plausible histories, per note/note.tex.

For a noise-free observation y = g(h) with g finite-to-one, the set of h values
compatible with y is the finite fiber H(y) = g^{-1}(y).  Choosing one element
for each observed point gives a *plausible history*; there are
prod_i |H(y(x_i))| of them, and the Model 3 posterior is a finite weighted sum
over that set.  No MCMC, no Monte Carlo, no BO loop -- which is exactly what
makes this useful as ground truth for the sampler in ``arms.py``.

What it provides:

  * the posterior weights P(H = H) over plausible histories;
  * the three CDFs the note asks to compare,
        Model 1  P(y(x) <= a | Y)  with a GP fitted to y,
        Model 2  P(y(x) <= a | H)  the true history,
        Model 3  P(y(x) <= a | Y)  the composite prior, Y only;
  * the Kolmogorov distance between them, which is zero exactly when the note
    predicts the distributions coincide;
  * the mutual information I(y(x); h(x_i)) of the note's last section;
  * a direct grading of the MCMC sampler against Model 3.

    python three_arm/exact.py                       # the comparison table
    python three_arm/exact.py --grade               # also grade the sampler
    python three_arm/exact.py --maps shiftsq_c0 abs --n 8

Only maps with finite fibers can be enumerated: identity, exp, sinh, the
shifted squares, abs and the cosines.  relu and clip have half-line fibers and
the two-dimensional analogs have continuous ones; those stay with the sampler.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.arms import KnownPrior, rbf                      # noqa: E402
from three_arm.test_functions import DTYPE, make_problem        # noqa: E402

SQRT2 = math.sqrt(2.0)
# Grid for integrating over h(x) at the query point.  The integrand is a 1-D
# Gaussian times an indicator, so a dense trapezoid over +/-12 sd is accurate
# to far beyond the precision anything here is compared at.
REACH = 8.0          # h is standardized; beyond 8 the prior carries ~1e-9


# --------------------------------------------------------------------------
# Fibers: g^{-1}(v) in *raw* units, before the problem's affine normalization
# --------------------------------------------------------------------------
def fiber(map_name: str, v: float, reach: float = 8.0):
    """Every h with outer.fn(h) == v, for the finite-fiber maps."""
    if map_name == "identity":
        return [v]
    if map_name.startswith("exp_b"):
        b = float(map_name[5:])
        return [math.log(v) / b] if v > 0 else []
    if map_name.startswith("sinh_b"):
        b = float(map_name[6:])
        return [math.asinh(v) / b]
    if map_name.startswith("shiftsq_c"):
        c = float(map_name[9:])
        if v < 0:
            return []
        r = math.sqrt(v)
        return [c] if r == 0.0 else [c + r, c - r]
    if map_name == "abs":
        return [] if v < 0 else ([0.0] if v == 0 else [v, -v])
    if map_name.startswith("cos_w"):
        w = float(map_name[5:])
        if abs(v) > 1.0:
            return []
        base = math.acos(max(-1.0, min(1.0, v)))
        out, period = [], 2.0 * math.pi / w
        kmax = int(reach / period) + 2
        for k in range(-kmax, kmax + 1):
            for z in ((base / w) + k * period, (-base / w) + k * period):
                if abs(z) <= reach:
                    out.append(z)
        return sorted(set(round(z, 12) for z in out))
    raise ValueError(f"{map_name} has no finite fiber")


def dg(map_name: str, z: float) -> float:
    """|d outer.fn / dh| at z -- the change-of-variables factor.

    For the even maps it is equal across the branches of a fiber and cancels,
    but it does not in general, so it is carried.
    """
    if map_name == "identity":
        return 1.0
    if map_name.startswith("exp_b"):
        b = float(map_name[5:]); return abs(b * math.exp(b * z))
    if map_name.startswith("sinh_b"):
        b = float(map_name[6:]); return abs(b * math.cosh(b * z))
    if map_name.startswith("shiftsq_c"):
        c = float(map_name[9:]); return abs(2.0 * (z - c))
    if map_name == "abs":
        return 1.0
    if map_name.startswith("cos_w"):
        w = float(map_name[5:]); return abs(w * math.sin(w * z))
    raise ValueError(map_name)


def fibers_for(problem, y: torch.Tensor):
    """Fibers at every observed point, in standardized h units."""
    span = problem.q99 - problem.q01
    out = []
    for yi in y.tolist():
        v = yi * span + problem.q01
        f = fiber(problem.name, v)
        if not f:
            raise RuntimeError(f"empty fiber at y={yi:.4f} on {problem.name}")
        out.append(f)
    return out


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------
def histories(problem, X, y, prior, max_histories=200_000):
    """All plausible histories with their posterior weights.

    P(H = H) is proportional to the GP prior density at that assignment,
    divided by the Jacobian of g at each chosen branch.
    """
    fibs = fibers_for(problem, y)
    total = 1
    for f in fibs:
        total *= len(f)
    if total > max_histories:
        raise RuntimeError(f"{total:,} histories; raise --max or lower --n")

    L = prior.unit_chol(X)
    m, v = float(prior.m[0]), float(prior.v[0])
    rows, logw = [], []
    for combo in itertools.product(*fibs):
        h = torch.tensor(combo, dtype=DTYPE)
        r = (h - m) / math.sqrt(v)                      # unit-scale residual
        a = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        lp = -0.5 * float(r @ a)                        # prior, up to a constant
        lj = sum(math.log(max(dg(problem.name, z), 1e-300)) for z in combo)
        rows.append(h); logw.append(lp - lj)
    H = torch.stack(rows)                               # (nH, n)
    lw = torch.tensor(logw, dtype=DTYPE)
    return H, torch.softmax(lw, 0), L


# --------------------------------------------------------------------------
# CDFs
# --------------------------------------------------------------------------
def _conditional(prior, X, Hmat, xq, L):
    """h(xq) | history, for every history at once: means (nH,) and one sd."""
    Ks = rbf(X, xq, prior.ls)                           # (n, 1)
    A = torch.cholesky_solve(Ks, L)
    m, v = float(prior.m[0]), float(prior.v[0])
    mean = m + (Hmat - m) @ A.squeeze(-1)
    var = v * float((1.0 - (Ks * A).sum()).clamp_min(1e-12))
    return mean, math.sqrt(var)


def _sublevel(problem, a: float, reach: float = REACH):
    """The set {z : g(z) <= a}, as a list of intervals.

    g is piecewise monotone, so the boundary of a sublevel set consists of
    preimages of a -- which the fiber routine already computes.  Sorting those
    and testing one interior point per gap gives the set exactly, so the CDF
    below is analytic rather than quadrature.  Outside +/-reach the Gaussian
    carries less than 1e-9 of its mass, so treating the two outer pieces as
    single intervals is harmless even for the oscillating maps.
    """
    span = problem.q99 - problem.q01
    v = a * span + problem.q01
    bs = sorted(fiber(problem.name, v, reach=reach))
    edges = [-reach - 1.0] + bs + [reach + 1.0]
    raw = lambda z: float(problem.outer.fn(torch.tensor([[z]], dtype=DTYPE)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if raw(0.5 * (lo + hi)) <= v:
            out.append((-math.inf if lo < -reach else lo,
                        math.inf if hi > reach else hi))
    return out


def _ncdf(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def _cdf_from_gaussian_h(problem, mean, sd, grid_a):
    """P(g(h) <= a) for h ~ N(mean, sd^2), exactly, for a vector of means.

    Returns (len(mean), len(grid_a)).  The sublevel set depends only on a, so
    it is computed once per threshold and reused across histories.
    """
    out = torch.zeros(len(mean), len(grid_a), dtype=DTYPE)
    for t, a in enumerate(grid_a.tolist()):
        for lo, hi in _sublevel(problem, a):
            upper = (torch.ones_like(mean) if hi == math.inf
                     else _ncdf((hi - mean) / sd))
            lower = (torch.zeros_like(mean) if lo == -math.inf
                     else _ncdf((lo - mean) / sd))
            out[:, t] += upper - lower
    return out.clamp(0.0, 1.0)


def model3_cdf(problem, prior, X, y, xq, grid_a):
    Hmat, wts, L = histories(problem, X, y, prior)
    mean, sd = _conditional(prior, X, Hmat, xq, L)
    per = _cdf_from_gaussian_h(problem, mean, sd, grid_a)             # (nH, A)
    return (wts.unsqueeze(-1) * per).sum(0), Hmat, wts


def model2_cdf(problem, prior, X, Htrue, xq, grid_a):
    L = prior.unit_chol(X)
    mean, sd = _conditional(prior, X, Htrue.unsqueeze(0), xq, L)
    return _cdf_from_gaussian_h(problem, mean, sd, grid_a)[0]


def model1_cdf_oracle(problem, prior, X, y, xq, grid_a):
    """Model 1 given the prior on y that the composite model *implies*.

    Only defined when g is affine in h, where a linear combination of
    independent GPs is itself a GP.  With the affine normalization folded in,
    f = (sum_j c_j h_j - q01) / span, so

        mean  = (sum_j c_j m_j - q01) / span,
        var   = (sum_j c_j^2 v_j) / span^2,

    and the kernel is the same RBF.  This is the comparison the note's linear
    conjecture is about: handing Model 1 a *fitted* kernel instead mixes the
    conjecture with hyperparameter estimation, which is a separate effect.
    """
    coefs = problem.outer.linear_coef
    if coefs is None and problem.outer.kind != "identity":
        return None
    c = coefs if coefs is not None else [1.0]
    span = problem.q99 - problem.q01
    m0 = (sum(ci * mi for ci, mi in zip(c, problem.prior_means)) - problem.q01) / span
    v0 = sum(ci * ci * vi for ci, vi in zip(c, problem.prior_vars)) / span ** 2

    K = rbf(X, X, prior.ls) + prior.jitter * torch.eye(len(X), dtype=DTYPE)
    L = torch.linalg.cholesky(K)
    ks = rbf(X, xq, prior.ls)
    a = torch.cholesky_solve(ks, L)
    mu = m0 + float((a.squeeze(-1) * (y - m0)).sum())
    var = v0 * float((1.0 - (ks * a).sum()).clamp_min(1e-12))
    return 0.5 * (1.0 + torch.erf((grid_a - mu) / (math.sqrt(var) * SQRT2)))


def model3_cdf_linear(problem, prior, X, y, xq, grid_a):
    """Model 3 for an affine g, where the posterior is Gaussian in closed form.

    The fiber is continuous so it cannot be enumerated, but it does not need to
    be: conditioning a jointly Gaussian vector on a linear functional of it is
    exact.  Observing y constrains only sum_j c_j h_j, which is all f depends
    on, so this reduces to ordinary GP regression on y -- which is the content
    of the conjecture.
    """
    return model1_cdf_oracle(problem, prior, X, y, xq, grid_a)


def model1_cdf(X, y, xq, grid_a):
    """GP fitted to y with maximum likelihood, as the direct arm does."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import warnings
    warnings.filterwarnings("ignore")
    gp = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    except Exception:
        gp.eval()
    with torch.no_grad():
        post = gp.posterior(xq)
        mu = float(post.mean.squeeze()); sd = float(post.variance.sqrt().squeeze())
    return 0.5 * (1.0 + torch.erf((grid_a - mu) / (sd * SQRT2)))


# --------------------------------------------------------------------------
# Mutual information  I(y(x); h(x_i))
# --------------------------------------------------------------------------
def mutual_information(problem, prior, X, y, xq, idx, grid_a):
    """Between the query value and which branch point x_idx took.

    In the noise-free finite-fiber setting h(x_i) is a discrete variable -- the
    branch -- so this is H(branch) - E_y[H(branch | y(x))], the bits about
    h(x_i) that an extra observation at xq would reveal.
    """
    Hmat, wts, L = histories(problem, X, y, prior)
    branch = Hmat[:, idx]
    labels = sorted(set(round(float(b), 10) for b in branch))
    if len(labels) < 2:
        return 0.0
    mean, sd = _conditional(prior, X, Hmat, xq, L)
    cdf = _cdf_from_gaussian_h(problem, mean, sd, grid_a)             # (nH, A)
    dens = torch.diff(cdf, dim=-1).clamp_min(0.0)                     # (nH, A-1)

    prior_p = torch.tensor([float(wts[[round(float(b), 10) == l
                                      for b in branch]].sum()) for l in labels],
                           dtype=DTYPE)
    ent = lambda p: float(-(p[p > 0] * p[p > 0].log()).sum()) / math.log(2)

    joint = torch.stack([(wts[[round(float(b), 10) == l for b in branch]].unsqueeze(-1)
                          * dens[[round(float(b), 10) == l for b in branch]]).sum(0)
                         for l in labels])                            # (L, A-1)
    py = joint.sum(0)
    keep = py > 1e-14
    cond = joint[:, keep] / py[keep]
    h_cond = float((py[keep] * torch.tensor(
        [ent(cond[:, t]) for t in range(int(keep.sum()))], dtype=DTYPE)).sum())
    return ent(prior_p) - h_cond


# --------------------------------------------------------------------------
def kolmogorov(a, b):
    return float((a - b).abs().max())


def _fit_n(problem, d, n, cap):
    """Largest m <= n whose history count fits under the cap.

    The count is prod_i |fiber(y_i)|, so it grows geometrically in the number
    of observations: the cosines have six to twenty preimages each and run out
    of room several points before the squares do.
    """
    from torch.quasirandom import SobolEngine
    full = SobolEngine(d, scramble=True, seed=11).draw(n).to(DTYPE)
    for m in range(n, 1, -1):
        total = 1
        for f in fibers_for(problem, problem.f(full[:m])):
            total *= len(f)
            if total > cap:
                break
        if total <= cap:
            return m
    return 2


def run(map_name, n, d, draw_seed, n_query, grade, noise, cap=60_000):
    from torch.quasirandom import SobolEngine
    p = make_problem(map_name, d=d, lengthscale=0.5, draw_seed=draw_seed)
    prior = KnownPrior.from_problem(p)
    if p.outer.k == 1 and p.outer.kind == "lossy":
        n = _fit_n(p, d, n, cap)
    X = SobolEngine(d, scramble=True, seed=11).draw(n).to(DTYPE)
    Htrue = p.h(X)[:, 0]
    y = p.f(X)
    Xq = SobolEngine(d, scramble=True, seed=77).draw(n_query).to(DTYPE)
    grid_a = torch.linspace(-0.6, 1.6, 441, dtype=DTYPE)

    affine = p.outer.linear_coef is not None or p.outer.kind == "identity"
    d13 = d23 = d1o = 0.0
    mi = []
    nH = None
    for q in range(n_query):
        xq = Xq[q:q + 1]
        if affine:
            # Continuous fiber, but the posterior is Gaussian in closed form.
            c3 = model3_cdf_linear(p, prior, X, y, xq, grid_a)
            Hmat, wts, nH = None, None, 1
        else:
            c3, Hmat, wts = model3_cdf(p, prior, X, y, xq, grid_a)
            nH = Hmat.shape[0]
        c2 = model2_cdf(p, prior, X, Htrue, xq, grid_a) if p.outer.k == 1 else None
        c1 = model1_cdf(X, y, xq, grid_a)
        d13 = max(d13, kolmogorov(c1, c3))
        if c2 is not None:
            d23 = max(d23, kolmogorov(c2, c3))
        c1o = model1_cdf_oracle(p, prior, X, y, xq, grid_a)
        if c1o is not None:
            d1o = max(d1o, kolmogorov(c1o, c3))
        mi.append(0.0 if affine
                  else mutual_information(p, prior, X, y, xq, 0, grid_a))

    row = dict(map=map_name, n=n, nH=nH, d13=d13, d23=d23,
               d1o=d1o if affine else float("nan"),
               mi=sum(mi) / len(mi), grade=None)

    if grade:
        from three_arm.arms import LatentArm
        arm = LatentArm(p, noise=noise, n_chains=4, burn_first=1500,
                        burn=1500, keep=4000, thin=5, shortcuts=False, seed=0)
        arm.fit(X, None, y)
        # Compare the sampler's branch weights with the exact ones.
        Hmat, wts, _ = histories(p, X, y, prior)
        labels = Hmat[:, 0]
        got = arm.F[:, 0, 0]
        uniq = sorted(set(round(float(b), 6) for b in labels))
        exact_p = [float(wts[[round(float(b), 6) == u for b in labels]].sum()) for u in uniq]
        samp_p = [float(((got - u).abs() < 0.25).to(DTYPE).mean()) for u in uniq]
        row["grade"] = max(abs(a - b) for a, b in zip(exact_p, samp_p))
        row["exact_p"] = exact_p
        row["samp_p"] = samp_p
    return row


ENUMERABLE = ["identity", "sum", "exp_b1", "sinh_b1", "shiftsq_c0", "shiftsq_c0.5",
              "shiftsq_c1", "shiftsq_c2", "shiftsq_c3", "abs",
              "cos_w1", "cos_w2", "cos_w4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=ENUMERABLE)
    ap.add_argument("--n", type=int, default=8, help="observed points (histories grow as |fiber|^n)")
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--draw-seed", type=int, default=0)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--grade", action="store_true", help="also grade the MCMC sampler")
    ap.add_argument("--noise", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    hdr = (f"{'map':13s} {'n':>3s} {'#hist':>7s} {'sup|M1-M3|':>11s} {'M1 oracle':>10s} "
           f"{'sup|M2-M3|':>11s} {'I(y;h) bits':>12s}")
    if a.grade:
        hdr += f" {'sampler err':>12s}"
    print(f"n={a.n} observed points, {a.queries} query points, draw {a.draw_seed}\n")
    print(hdr); print("-" * len(hdr))
    for m in a.maps:
        try:
            r = run(m, a.n, a.d, a.draw_seed, a.queries, a.grade, a.noise)
        except RuntimeError as exc:
            print(f"{m:13s}  {exc}")
            continue
        orc = "    -     " if r['d1o'] != r['d1o'] else f"{r['d1o']:10.6f}"
        line = (f"{r['map']:13s} {r['n']:3d} {r['nH']:7d} {r['d13']:11.4f} {orc} "
                f"{r['d23']:11.4f} {r['mi']:12.4f}")
        if a.grade:
            line += f" {r['grade']:12.4f}"
        print(line, flush=True)
    print("\nsup|M1-M3|  direct GP, hyperparameters by MLE, vs the composite-prior posterior")
    print("M1 oracle   the same with Model 1 given the prior the composite model implies;")
    print("            defined only for affine g, and 0 iff the linear conjecture holds")
    print("sup|M2-M3|  observing h vs not; 0 iff the bijection result holds")
    print("I(y;h)      bits about the branch at x_1 revealed by one observation at x")
    print("n           reduced per map so the history count stays enumerable")


if __name__ == "__main__":
    main()
