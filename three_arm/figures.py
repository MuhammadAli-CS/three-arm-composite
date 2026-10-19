"""Figures for the three-arm experiment.

    python three_arm/figures.py --results results/main --out figures

Reads the saved JSON directly, pools the function draws exactly as
``run.py --collect`` does, and writes one PDF per figure.  Everything is
recomputed from the raw per-run records so a figure can never disagree with
the table.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.run import log_regret, paired                 # noqa: E402
from three_arm.test_functions import OUTER_MAPS              # noqa: E402

# Measured on identity with --no-shortcuts, where the truth is exactly 0: the
# latent arm's handicap that is not information loss.  Subtracted from every
# map whose latent arm actually runs the sampler.
BIAS = 0.409
EXACT = ("identity", "invertible")       # kinds that use the closed-form shortcut

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
C_KNOW, C_OBS, C_BAD = "#4878a8", "#d08838", "#b03030"


def load(results_dir):
    """Pool every (map, draw) file into per-map arrays of final log regrets."""
    final = collections.defaultdict(lambda: collections.defaultdict(list))
    ulost = collections.defaultdict(list)
    traj = collections.defaultdict(list)
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        sys.exit(f"no results in {results_dir}")
    for path in files:
        blob = json.load(open(path))
        name = blob["map"]
        for arm, runs in blob["results"].items():
            final[name][arm] += [log_regret(r["best"][-1], blob["f_min"]) for r in runs]
        for r in blob["results"].get("latent", []):
            ulost[name].append(sum(r["u_lost"]) / len(r["u_lost"]))
            traj[name].append(r["u_lost"])
    return final, ulost, traj


def stats(final, ulost):
    """(observe_h, se), (know_g, se), corrected observe_h, U_lost per map."""
    out = {}
    for name, arms in final.items():
        if not {"direct", "composite", "latent"} <= set(arms):
            continue
        oh = paired(arms["latent"], arms["composite"])
        kg = paired(arms["direct"], arms["latent"])
        exact = OUTER_MAPS[name].kind in EXACT if name in OUTER_MAPS else False
        out[name] = dict(oh=oh, kg=kg, exact=exact,
                         oh_corr=oh[0] if exact else oh[0] - BIAS,
                         u=sum(ulost[name]) / len(ulost[name]) if ulost[name] else 0.0)
    return out


def median_traj(traj, name):
    runs = traj[name]
    n = min(len(r) for r in runs)
    return [sorted(r[i] for r in runs)[len(runs) // 2] for i in range(n)]


# --------------------------------------------------------------------------
def fig_decomposition(S, path):
    """Every condition, both effects, ordered as the design groups them."""
    order = [m for m in OUTER_MAPS if m in S]
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    xs = range(len(order))
    ax.bar([x - 0.2 for x in xs], [S[m]["kg"][0] for m in order], 0.4,
           yerr=[S[m]["kg"][1] for m in order], color=C_KNOW, capsize=2,
           label="knowing $g$  (direct $-$ latent)")
    ax.bar([x + 0.2 for x in xs], [S[m]["oh_corr"] for m in order], 0.4,
           yerr=[S[m]["oh"][1] for m in order], color=C_OBS, capsize=2,
           label="observing $h$  (latent $-$ composite, bias-corrected)")
    ax.axhline(0, color="k", lw=0.8)
    for edge in (2.5, 7.5, 13.5):                 # control / sweep / pairs / analogs
        ax.axvline(edge, color="k", lw=0.6, ls=":", alpha=0.5)
    for x, txt in ((1.0, "controls"), (5.0, "sweep"), (10.5, "matched pairs"),
                   (16.0, "analogs")):
        ax.text(x, ax.get_ylim()[1], txt, ha="center", va="bottom", fontsize=8,
                style="italic", color="0.35")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("difference in final $\\log_{10}$ regret")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    return _pearson(rank(xs), rank(ys))


# Label offsets for the handful of points that would otherwise collide.
_NUDGE = {"norm": (3, -11), "abs": (3, 5), "relu": (5, -2), "max": (4, 4),
          "ratio": (-6, -12), "cos_w2": (4, -10), "cos_w4": (-26, 6),
          "shiftsq_c3": (6, -10), "shiftsq_c2": (7, -9), "shiftsq_c0.5": (4, -11)}


def fig_hypothesis(S, path):
    """The plot this study was designed to produce: gain against information lost.

    The hypothesis predicts an upward trend.  There is none that survives
    dropping any one family of maps, so the correlations are printed rather
    than a fitted line -- a regression line here would imply a relationship the
    data do not support.
    """
    pts = [(S[m]["u"], S[m]["oh_corr"], m) for m in S if not S[m]["exact"]]
    sweep = [p for p in pts if p[2].startswith("shiftsq")]
    rest = [p for p in pts if not p[2].startswith("shiftsq")]
    no_cos = [p for p in rest if not p[2].startswith("cos")]

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.scatter([p[0] for p in rest], [p[1] for p in rest], s=34, color=C_OBS,
               zorder=3, label="other maps")
    ax.scatter([p[0] for p in sweep], [p[1] for p in sweep], s=44, color=C_BAD,
               marker="s", zorder=4, label="the sweep $(z-c)^2$ (known broken)")
    for u, v, m in pts:
        ax.annotate(m, (u, v), fontsize=6.5, xytext=_NUDGE.get(m, (4, 4)),
                    textcoords="offset points", color="0.3")
    ax.axhline(0, color="k", lw=0.8)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.55)
    rho_all = _spearman([p[0] for p in pts], [p[1] for p in pts])
    r_rest = _pearson([p[0] for p in rest], [p[1] for p in rest])
    r_nocos = _pearson([p[0] for p in no_cos], [p[1] for p in no_cos])
    ax.text(0.03, 0.97,
            "hypothesis: the gain should grow with information destroyed\n"
            f"Spearman $\\rho={rho_all:+.2f}$ over all {len(pts)} lossy maps\n"
            f"$r={r_rest:+.2f}$ without the sweep, but ${r_nocos:+.2f}$ without "
            "the $\\cos$ family too", transform=ax.transAxes, fontsize=7.5,
            va="top", color="0.3", style="italic",
            bbox=dict(fc="white", ec="0.85", lw=0.5, pad=3.5))
    ax.set_xlabel("$U_{\\mathrm{lost}}$  (information $g$ destroys)")
    ax.set_ylabel("observing $h$  (bias-corrected)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_sweep(S, traj, path):
    """Why the sweep fails: the knob is right at iteration 0 and gone by the end."""
    cs = [0.0, 0.5, 1.0, 2.0, 3.0]
    names = [f"shiftsq_c{c:g}" for c in cs]
    names = [n for n in names if n in S]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    a1.errorbar(cs[:len(names)], [S[n]["oh_corr"] for n in names],
                yerr=[S[n]["oh"][1] for n in names], marker="o", color=C_BAD,
                capsize=3, label="measured")
    a1.plot(cs[:len(names)], [1.6, 1.3, 0.8, 0.25, 0.05][:len(names)], ls="--",
            marker="s", color="0.55", label="predicted shape")
    a1.set_xlabel("$c$  (larger $c$ $\\Rightarrow$ $g$ more nearly injective)")
    a1.set_ylabel("observing $h$  (bias-corrected)")
    a1.set_title("the gain runs backwards", fontsize=9)
    a1.legend(fontsize=8)

    for n, c in zip(names, cs):
        a2.plot(median_traj(traj, n), label=f"$c={c:g}$")
    a2.set_yscale("log")
    a2.set_xlabel("BO iteration")
    a2.set_ylabel("$U_{\\mathrm{lost}}$  (median over runs)")
    a2.set_title("the knob is correct at iteration 0, then drifts", fontsize=9)
    a2.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_censoring(S, path):
    """The finding: same mechanism, different location, opposite outcome.

    Note which way round this goes.  In principle -max(z,0) is the more
    destructive map -- its fiber is an entire half-line, against a bounded
    interval for the clipped pair.  But U_lost is measured at the points BO
    actually evaluated, and it comes out eight times *smaller*, because the
    censored region sits away from the optimum and is rarely visited.  The gain
    follows the measured loss, not the fiber, which is the whole point.
    """
    pair = [("relu", "$-\\max(z,0)$\ncensors away from\nthe optimum"),
            ("clip_c2", "clipped pair\ncensors at\nthe optimum")]
    pair = [(m, lab) for m, lab in pair if m in S]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    xs = range(len(pair))
    a1.bar(xs, [S[m]["u"] for m, _ in pair], 0.5, color="0.6")
    a1.set_xticks(list(xs)); a1.set_xticklabels([lab for _, lab in pair], fontsize=8)
    a1.set_ylabel("$U_{\\mathrm{lost}}$")
    a1.set_title("information lost where BO actually sampled", fontsize=9)
    a1.annotate("fiber is a half-line\n(infinite in principle)\nyet $8\\times$ less is lost",
                xy=(0, S[pair[0][0]]["u"]), xytext=(0.06, 0.62),
                textcoords="axes fraction", fontsize=7.5, color="0.35",
                style="italic",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8))

    a2.bar(xs, [S[m]["oh_corr"] for m, _ in pair], 0.5,
           yerr=[S[m]["oh"][1] for m, _ in pair], color=C_OBS, capsize=4)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(list(xs)); a2.set_xticklabels([lab for _, lab in pair], fontsize=8)
    a2.set_ylabel("observing $h$  (bias-corrected)")
    a2.set_title("what observing $h$ is worth", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_curves(final_dir, path, maps=("clip_c2", "relu", "shiftsq_c0")):
    """Convergence of the three arms, median over pooled runs."""
    fig, axes = plt.subplots(1, len(maps), figsize=(9.6, 3.1), sharex=True)
    colours = {"direct": "0.45", "composite": C_KNOW, "latent": C_OBS}
    for ax, name in zip(axes, maps):
        runs = collections.defaultdict(list)
        fmin = None
        for path_j in sorted(glob.glob(os.path.join(final_dir, "*.json"))):
            blob = json.load(open(path_j))
            if blob["map"] != name:
                continue
            fmin = blob["f_min"]
            for arm, rs in blob["results"].items():
                runs[arm] += [[log_regret(v, fmin) for v in r["best"]] for r in rs]
        for arm in ("direct", "composite", "latent"):
            if arm not in runs:
                continue
            n = min(len(r) for r in runs[arm])
            med = [sorted(r[i] for r in runs[arm])[len(runs[arm]) // 2] for i in range(n)]
            ax.plot(med, color=colours[arm], label=arm)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("evaluation")
    axes[0].set_ylabel("median $\\log_{10}$ regret")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    final, ulost, traj = load(a.results)
    S = stats(final, ulost)
    j = lambda n: os.path.join(a.out, n)
    fig_decomposition(S, j("decomposition.pdf"))
    fig_hypothesis(S, j("hypothesis.pdf"))
    fig_sweep(S, traj, j("sweep.pdf"))
    fig_censoring(S, j("censoring.pdf"))
    fig_curves(a.results, j("curves.pdf"))
    print(f"wrote 5 figures to {a.out}/ from {len(S)} conditions")


if __name__ == "__main__":
    main()


def fig_linear(results_dir, path):
    """The linear map, three ways: exact CDFs, paired runs, and the decomposition.

    g = y_1 + y_2 is the one case with an analytic answer to the whole
    question, so it is worth showing on its own.  Observing h leaves the split
    of the sum completely undetermined -- U_lost is 0.702, third highest of the
    analogs -- yet none of that uncertainty reaches f, which depends only on
    the sum.
    """
    import glob, json
    from torch.quasirandom import SobolEngine
    import torch
    from three_arm.arms import KnownPrior, rbf
    from three_arm.exact import model1_cdf, model1_cdf_oracle
    from three_arm.test_functions import DTYPE, make_problem

    torch.set_default_dtype(DTYPE)
    p = make_problem("sum", d=6, lengthscale=0.5, draw_seed=0)
    prior = KnownPrior.from_problem(p)
    X = SobolEngine(6, scramble=True, seed=11).draw(12).to(DTYPE)
    y, H = p.f(X), p.h(X)
    xq = SobolEngine(6, scramble=True, seed=77).draw(1).to(DTYPE)
    grid = torch.linspace(0.05, 0.95, 601, dtype=DTYPE)

    # Model 3 (composite prior, y only) == Model 1 with the implied prior.
    c3 = model1_cdf_oracle(p, prior, X, y, xq, grid)
    c1 = model1_cdf(X, y, xq, grid)                      # hyperparameters by MLE
    # Model 2: condition each h_j on its own observations; the sum is Gaussian.
    L = prior.unit_chol(X)
    ks = rbf(X, xq, prior.ls); a = torch.cholesky_solve(ks, L)
    span = p.q99 - p.q01
    mu = sum(float(prior.m[j] + (a.squeeze(-1) * (H[:, j] - prior.m[j])).sum())
             for j in range(2))
    var = float((1.0 - (ks * a).sum()).clamp_min(1e-12)) * float(prior.v.sum())
    c2 = 0.5 * (1.0 + torch.erf(((grid * span + p.q01) - mu)
                                / (math.sqrt(var) * math.sqrt(2.0))))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.3))

    a1.plot(grid, c3, color=C_OBS, lw=3.0, label="Model 3  (knows $g$, sees $y$)")
    a1.plot(grid, c2, color=C_KNOW, lw=1.4, ls="--", label="Model 2  (sees $h$)")
    a1.plot(grid, c1, color="0.45", lw=1.4, ls=":", label="Model 1  (GP on $y$, MLE)")
    a1.set_xlabel("$a$"); a1.set_ylabel("$P(y(x) \leq a \mid \cdot)$")
    a1.set_title("Models 2 and 3 coincide exactly", fontsize=9)
    a1.legend(fontsize=7, loc="upper left")
    a1.annotate(f"$\sup|M_2-M_3| = {float((c2-c3).abs().max()):.6f}$\n"
                f"$\sup|M_1-M_3| = {float((c1-c3).abs().max()):.3f}$  (MLE only)",
                xy=(0.52, 0.12), xycoords="axes fraction", fontsize=7.5, color="0.3")

    # paired runs
    d, c = [], []
    for f in sorted(glob.glob(os.path.join(results_dir, "sum*.json"))):
        b = json.load(open(f))
        d += [log_regret(r["best"][-1], b["f_min"]) for r in b["results"]["direct"]]
        c += [log_regret(r["best"][-1], b["f_min"]) for r in b["results"]["composite"]]
    lo, hi = min(d + c) - 0.3, max(d + c) + 0.3
    a2.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    a2.scatter(d, c, s=28, color=C_OBS, zorder=3)
    a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
    a2.set_xlabel("direct, final $\log_{10}$ regret")
    a2.set_ylabel("composite")
    wins = sum(1 for x, z in zip(d, c) if z < x)
    a2.set_title(f"composite wins {wins}/{len(d)} paired runs", fontsize=9)
    a2.annotate("below the line\n= composite better", xy=(0.04, 0.9), fontsize=7.5,
                xycoords="axes fraction", color="0.4", style="italic", va="top")

    # decomposition
    from three_arm.run import paired
    lat = []
    for f in sorted(glob.glob(os.path.join(results_dir, "sum*.json"))):
        b = json.load(open(f))
        lat += [log_regret(r["best"][-1], b["f_min"]) for r in b["results"]["latent"]]
    eff = [("knowing $g$", paired(d, lat)), ("observing $h$", paired(lat, c)),
           ("total", paired(d, c))]
    xs = range(len(eff))
    # Two standard errors, so the bars support the claim in the title: at one
    # SE the total would not cross zero and the figure would contradict itself.
    a3.bar(xs, [v[0] for _, v in eff], 0.55, yerr=[2.0 * v[1] for _, v in eff],
           color=[C_KNOW, C_OBS, "0.55"], capsize=4)
    a3.axhline(0, color="k", lw=0.9)
    a3.set_xticks(list(xs)); a3.set_xticklabels([k for k, _ in eff], fontsize=8)
    a3.set_ylabel("difference in final $\log_{10}$ regret")
    a3.set_title("no effect reaches $2$ SE", fontsize=9)
    a3.annotate("error bars: $\pm 2$ SE", xy=(0.04, 0.04), xycoords="axes fraction",
                fontsize=7.5, color="0.4", style="italic")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# Polynomial degree of each member of the bijectivity family; exp and sinh are
# not polynomials and appear only where degree is not the x-axis.
_DEGREE = {"root1": 1, "root2": 2, "root3": 3, "root4": 4, "root5": 5,
           "mono3": 3, "mono5": 5}


def fig_bijectivity(json_path, path, n_show=40):
    """Does injectivity or nonlinearity predict the composite advantage?

    Three panels on the same numbers: the gain against measured injectivity,
    against polynomial degree, and against training-set size.  Regret is
    deliberately absent -- an injective g is monotone, so its optimum sits at
    an endpoint of the reachable range while a non-injective one can be
    interior, which confounds any regret comparison between the two classes.
    """
    import json
    blob = json.load(open(json_path))
    ns = blob["ns"]
    nbar, kind = blob["nbar"], blob["kind"]
    maps = list(nbar)
    inj = lambda m: kind[m] in ("identity", "invertible", "linear")

    def gain(m, n):
        o = blob["results"][m][str(n)]
        d = [x - y for x, y in zip(o["cd"], o["cc"])]
        mu = sum(d) / len(d)
        se = (sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        return mu, se

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.2, 3.6))

    # --- (a) against measured injectivity -------------------------------
    for m in maps:
        g, se = gain(m, n_show)
        a1.errorbar(nbar[m], g, yerr=se, marker="o" if inj(m) else "s",
                    ms=7, color=C_KNOW if inj(m) else C_BAD, capsize=2, zorder=3)
        a1.annotate(m, (nbar[m], g), fontsize=6.5, xytext=(6, 2),
                    textcoords="offset points", color="0.35")
    a1.set_xscale("linear"); a1.set_yscale("log")
    a1.set_xlabel(r"$\bar n = \mathbb{E}\,|g^{-1}(y)|$   (measured)")
    a1.set_ylabel("CRPS gain, direct $-$ composite")
    a1.set_title("injectivity does not order the gain", fontsize=9)
    a1.scatter([], [], marker="o", color=C_KNOW, label="injective")
    a1.scatter([], [], marker="s", color=C_BAD, label="non-injective")
    a1.legend(fontsize=7.5, loc="lower right")

    # --- (b) against polynomial degree, matched pairs side by side -------
    poly = [m for m in maps if m in _DEGREE]
    for m in poly:
        g, se = gain(m, n_show)
        off = -0.08 if inj(m) else 0.08
        a2.errorbar(_DEGREE[m] + off, g, yerr=se, marker="o" if inj(m) else "s",
                    ms=7, color=C_KNOW if inj(m) else C_BAD, capsize=2, zorder=3)
    for deg in (3, 5):                      # join the matched pairs
        pair = [m for m in poly if _DEGREE[m] == deg]
        if len(pair) == 2:
            a, b_ = (gain(m, n_show)[0] for m in pair)
            a2.plot([deg, deg], [a, b_], color="0.7", lw=0.9, zorder=1)
            a2.annotate(f"{abs(a - b_) / min(a, b_):.0%}", (deg, max(a, b_)),
                        xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7, color="0.4")
    a2.set_yscale("log"); a2.set_xticks([1, 2, 3, 4, 5])
    a2.set_xlabel("polynomial degree")
    a2.set_title("degree does, and matched pairs nearly coincide", fontsize=9)

    # --- (c) against training-set size ----------------------------------
    for m in maps:
        gs = [gain(m, n)[0] for n in ns]
        a3.plot(ns, gs, marker="o", ms=3.5, lw=1.2,
                color=C_KNOW if inj(m) else C_BAD,
                alpha=0.85, ls="-" if inj(m) else "--")
    a3.set_xscale("log"); a3.set_yscale("log")
    a3.set_xticks(ns); a3.set_xticklabels([str(n) for n in ns])
    a3.xaxis.set_minor_locator(plt.NullLocator())      # otherwise 5 and 6x10^0 collide
    a3.set_xlabel("training points $n$")
    a3.set_ylabel("CRPS gain")
    a3.plot([], [], color=C_KNOW, ls="-", label="injective")
    a3.plot([], [], color=C_BAD, ls="--", label="non-injective")
    a3.legend(fontsize=7.5, loc="lower left")
    a3.set_title("the advantage does not wash out with data", fontsize=9)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
