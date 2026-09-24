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
