import json, io, os, glob, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES = "D:/SURP/MOBO-HighDim/composite_ablation/results/two_stage_matched"
OUT = "D:/SURP/three-arm-composite/note"

BIJ = ["identity", "exp_b1", "sinh_b1", "mono3", "mono5"]
NON = ["shiftsq_c0", "shiftsq_c0.5", "shiftsq_c1", "shiftsq_c2", "shiftsq_c3",
       "root2", "root3", "root4", "root5", "cos_w1", "cos_w2",
       "spr3s050", "spr4s050", "spr6s100"]
LABEL = {
    "identity": r"Identity $h$", "exp_b1": r"Exponential $e^{h}$",
    "sinh_b1": r"Hyperbolic sine", "mono3": r"Cubic $h^{3}$",
    "mono5": r"Quintic $h^{5}$",
    "shiftsq_c0": r"Square $h^{2}$",
    "shiftsq_c0.5": r"Shifted sq. $(h-0.5)^{2}$",
    "shiftsq_c1": r"Shifted sq. $(h-1)^{2}$",
    "shiftsq_c2": r"Shifted sq. $(h-2)^{2}$",
    "shiftsq_c3": r"Shifted sq. $(h-3)^{2}$",
    "root2": r"Two-root quadratic", "root3": r"Three-root cubic",
    "root4": r"Four-root quartic", "root5": r"Five-root quintic",
    "cos_w1": r"Cosine $\cos h$", "cos_w2": r"Cosine $\cos 2h$",
    "spr3s050": r"Three-root spread $0.5$",
    "spr4s050": r"Four-root spread $0.5$",
    "spr6s100": r"Six-root spread $1.0$",
}

def load(name):
    """Returns direct (S,T), composite (S,T) final-trace arrays, runs ordered
    draw-major exactly as the appendix tables order them."""
    D, C = [], []
    for d in range(4):
        p = os.path.join(RES, f"{name}_d{d}.json")
        r = json.load(io.open(p, encoding="utf-8"))["results"][name]
        D.extend(r["direct_trace"]); C.extend(r["composite_trace"])
    return np.array(D, float), np.array(C, float)

def get(name):
    D, C = load(name)
    if name == "root2":                     # pool the duplicate map, subsample to 20
        D2, C2 = load("spr2s050")
        D, C = np.vstack([D, D2]), np.vstack([C, C2])
        rnd = random.Random(0); keep = []
        for d in range(4):
            idx = list(range(5*d, 5*d+5)) + list(range(20+5*d, 25+5*d))
            keep.extend(rnd.sample(idx, 5))
        D, C = D[keep], C[keep]
    assert D.shape == C.shape and D.shape[0] == 20, (name, D.shape)
    return D, C

DATA = {m: get(m) for m in BIJ + NON}
T = DATA["identity"][0].shape[1]
EV = np.arange(T) + 10                      # n_init = 10, then 40 BO iterations

# ---------------------------------------------------------------- figure 1
# Final-HV gain per map, 95% CI, ordered within each bijectivity group.
def stats_for(m):
    D, C = DATA[m]
    pr = 100.0 * (C[:, -1] - D[:, -1]) / D[:, -1]
    n = len(pr); mu = pr.mean(); se = pr.std(ddof=1) / np.sqrt(n)
    t = mu / se if se > 0 else 0.0
    p = 1.0 - stats.t.cdf(t, n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return mu, h, p

S = {m: stats_for(m) for m in BIJ + NON}
ob = sorted(BIJ, key=lambda m: S[m][0])
on = sorted(NON, key=lambda m: S[m][0])
order = on + [None] + ob                    # non-bijective at the bottom of the axis

fig, ax = plt.subplots(figsize=(6.4, 6.2))
ypos, ylab, seen = [], [], 0
for i, m in enumerate(order):
    y = i
    if m is None:
        ypos.append(y); ylab.append("")
        continue
    mu, h, p = S[m]
    sig = p < 0.05
    col = "#1b5e9c" if m in NON else "#7a7a7a"
    ax.errorbar(mu, y, xerr=h, fmt="o", ms=5.5, lw=1.4, capsize=2.8,
                color=col, mfc=col if sig else "white", mew=1.4, zorder=3)
    ypos.append(y)
    ylab.append(LABEL[m] + ("*" if sig else ""))
ax.axvline(0, color="k", lw=0.9, ls="--", zorder=1)
ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=8.5)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel("Final-HV gain (\%)" if False else "Final-HV gain (%)", fontsize=10)
ax.text(0.985, 0.985, "bijective", transform=ax.transAxes, ha="right",
        va="top", fontsize=9, color="#555555")
ax.text(0.985, 0.012, "not bijective", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="#1b5e9c")
ax.axhline(len(on), color="#cccccc", lw=0.8)

# five-root quintic runs off the useful scale; clip and annotate it
XMAX = 12.0
mu5, h5, _ = S["root5"]
y5 = order.index("root5")
ax.annotate("", xy=(XMAX - 0.1, y5), xytext=(XMAX - 2.0, y5),
            ha="right", va="center", fontsize=8, color="#1b5e9c",
            arrowprops=dict(arrowstyle="-|>", color="#1b5e9c", lw=1.3,
                            shrinkA=0, shrinkB=0))
ax.text(XMAX - 2.3, y5 + 0.02, r"$+31.2$  [$4.0$, $58.5$]", ha="right",
        va="center", fontsize=8, color="#1b5e9c")
ax.set_xlim(-6.5, XMAX)
ax.grid(axis="x", lw=0.4, color="#e6e6e6", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hv_gain.pdf"))
fig.savefig(os.path.join(OUT, "fig_hv_gain.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Mean HV trace per arm.  No band: the run-to-run spread is dominated by the
# four function draws, whose hypervolumes differ by roughly an order of
# magnitude, and that spread cancels in the pairing.  The shaded wedge is the
# gap between the two means, which is what the paired test is about.
grid = BIJ + [None] + NON
nrow, ncol = 4, 5

def panel_grid(fname, kind):
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.0, 7.6), sharex=True)
    axf = axes.ravel()
    for ax, m in zip(axf, grid):
        if m is None:
            ax.axis("off"); continue
        D, C = DATA[m]
        p_ = S[m][2]
        if kind == "abs":
            md, mc = D.mean(0), C.mean(0)
            ax.plot(EV, md, color="#b03030", lw=1.6, label="Direct", zorder=3)
            ax.plot(EV, mc, color="#1b5e9c", lw=1.6, label="Composite", zorder=3)
            ax.fill_between(EV, md, mc, where=mc >= md, color="#1b5e9c",
                            alpha=0.22, lw=0, zorder=2)
            ax.fill_between(EV, md, mc, where=mc < md, color="#b03030",
                            alpha=0.22, lw=0, zorder=2)
        else:
            d = C - D                                  # paired, per run
            mu = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(d.shape[0])
            ax.axhline(0, color="k", lw=0.9, ls="--", zorder=1)
            ax.plot(EV, mu, color="#1b5e9c", lw=1.6, zorder=3)
            ax.fill_between(EV, mu - se, mu + se, color="#1b5e9c",
                            alpha=0.22, lw=0, zorder=2)
        ttl = LABEL[m] + chr(10) + (r"$p=%.3f$" % p_ if p_ >= 0.001 else r"$p<0.001$")
        ax.set_title(ttl, fontsize=8.2, pad=3, linespacing=1.25,
                     color="#1b5e9c" if p_ < 0.05 else "black")
        ax.tick_params(labelsize=7.5)
        ax.grid(lw=0.4, color="#ececec"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axf[len(grid):]:
        ax.axis("off")
    if kind == "abs":
        axf[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.supxlabel("function evaluations", fontsize=10)
    fig.supylabel("hypervolume" if kind == "abs"
                  else "hypervolume, Composite $-$ Direct", fontsize=10)
    fig.text(0.015, 0.985, "bijective", fontsize=9.5, color="#555555", va="top")
    fig.text(0.015, 0.735, "not bijective", fontsize=9.5, color="#1b5e9c", va="top")
    fig.tight_layout(rect=(0.035, 0.025, 1, 1))
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=200)
    plt.close(fig)

panel_grid("fig_hv_traces", "abs")
panel_grid("fig_hv_delta", "delta")

print("%-26s %8s %8s %8s %8s" % ("map", "direct", "comp", "gain%", "p"))
for m in BIJ + NON:
    D, C = DATA[m]; mu, h, p = S[m]
    print("%-26s %8.2f %8.2f %+8.2f %8.3f" % (m, D[:, -1].mean(), C[:, -1].mean(), mu, p))
