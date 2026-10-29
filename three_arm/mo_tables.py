"""Multi-objective tables and the matched single-vs-multi comparison.

Reads the two-objective grid, which lives in the sibling MOBO-HighDim
checkout because it depends on that repo's solvers, and emits the LaTeX for
note/problems.tex plus the scatter figure.

    python three_arm/mo_tables.py

Writes mo_table.tex, matched_table.tex and figures/so_vs_mo.pdf.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.run import log_regret                               # noqa: E402
from three_arm.tables import DTYPE, fmt_p, paired_t, tex           # noqa: E402

# Bijectivity over the range h reaches, matching the single-objective tables.
# shiftsq_c3 is NOT here: (h-3)^2 has its vertex strictly inside that range,
# so it is two-to-one there, even though the fold is sampled by only 0.12% of
# draws and nbar therefore reads 1.01.  See three_arm.tables.is_injective.
BIJECTIVE = {"identity", "exp_b1", "sinh_b1", "mono3", "mono5"}
DEFERRED = {"cos_w4"}


def load_mo(d):
    """map -> list of per-run percentage hypervolume differences."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = json.load(open(fp))
        for name, r in b["results"].items():
            out.setdefault(name, []).extend(
                100.0 * (c - dv) / abs(dv)
                for dv, c in zip(r["direct"], r["composite"]))
    return out


def load_traces(d, exclude=("cos_w4",)):
    """map -> (direct, composite), each a list of per-run hypervolume traces.

    The endpoint loader above throws the traces away; the convergence figure
    needs them.  Pooled over the per-(g1, draw) job files exactly as the
    endpoints are, so a map's curve and its table row come from the same runs.
    """
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = json.load(open(fp))
        for name, r in b["results"].items():
            if name in exclude:
                continue
            d_, c_ = out.setdefault(name, ([], []))
            d_.extend(r["direct_trace"])
            c_.extend(r["composite_trace"])
    return out


def load_so(d):
    """map -> list of per-run differences in final log10 regret."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = json.load(open(fp))
        per = {a: [log_regret(x["best"][-1], b["f_min"]) for x in rs]
               for a, rs in b["results"].items()}
        out.setdefault(b["map"], []).extend(
            x - y for x, y in zip(per["direct"], per["composite_mle"]))
    return out


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
            / math.sqrt(sum((x - ma) ** 2 for x in a)
                        * sum((y - mb) ** 2 for y in b)))


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    return pearson([ra[x] for x in a], [rb[y] for y in b])


def write_mo_table(mo, out):
    rows = []
    for m, v in mo.items():
        mean, sd, p = paired_t(v)
        rows.append((m, m in BIJECTIVE, mean, sd, p))
    rows.sort(key=lambda r: (not r[1], -r[2]))
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Two-objective grid: 21 outer maps, $4$ function draws "
             r"$\times$ $5$ initial designs. Gain is the per-run percentage "
             r"difference in final hypervolume, paired over the shared initial "
             r"design; $p$ is one-sided for a gain above zero. The second "
             r"objective is the identity, so both arms model it with the same "
             r"GP and every difference is attributable to $g_1$.}",
             r"\label{tab:mo}",
             r"\begin{tabular}{l r r r}", r"\toprule",
             r"Problem & Gain (\%) & $p$ & \\", r"\midrule"]
    last = None
    for m, bij, mean, sd, p in rows:
        if bij != last:
            lines.append(r"\multicolumn{4}{@{}l}{\itshape "
                         + ("Bijective" if bij else "Not bijective") + r"} \\")
            last = bij
        note = r"deferred" if m in DEFERRED else ""
        lines.append(f"{tex(m)} & ${mean:+.2f}\\pm{sd:.2f}$ & {fmt_p(p)} & {note} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(out, "mo_table.tex"), "w").write("\n".join(lines))
    print(f"wrote mo_table.tex ({len(rows)} rows)")
    return rows


def write_matched_table(so, mo, out):
    shared = sorted(set(so) & set(mo), key=lambda m: -paired_t(so[m])[0])
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{The $17$ maps measured in both settings. Single-objective "
             r"gain is in $\log_{10}$ regret, multi-objective in percent "
             r"hypervolume; the two are on different scales, so only the "
             r"\emph{ordering} is comparable. Rows are sorted by the "
             r"single-objective gain, and the multi-objective column does not "
             r"follow it.}",
             r"\label{tab:matched}",
             r"\begin{tabular}{l c r r r r}", r"\toprule",
             r" & & \multicolumn{2}{c}{Single objective} & "
             r"\multicolumn{2}{c}{Multi objective} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Problem & Bij.\ & Gain & $p$ & Gain (\%) & $p$ \\", r"\midrule"]
    S, M = [], []
    for m in shared:
        s_m, _, s_p = paired_t(so[m])
        o_m, _, o_p = paired_t(mo[m])
        S.append(s_m)
        M.append(o_m)
        lines.append(f"{tex(m)} & {'yes' if m in BIJECTIVE else 'no'} & "
                     f"${s_m:+.3f}$ & {fmt_p(s_p)} & ${o_m:+.2f}$ & {fmt_p(o_p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(out, "matched_table.tex"), "w").write("\n".join(lines))
    print(f"wrote matched_table.tex ({len(shared)} rows)")
    keep = [i for i, m in enumerate(shared) if m != "root5"]
    return dict(n=len(shared), shared=shared, S=S, M=M,
                r=pearson(S, M), rho=spearman(S, M),
                r_no5=pearson([S[i] for i in keep], [M[i] for i in keep]),
                rho_no5=spearman([S[i] for i in keep], [M[i] for i in keep]))


def write_figure(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    # root5 gains +31% against a bulk spanning -3 to +6, so plotting it to
    # scale compresses every other map into a strip.  Clip the axis to the
    # bulk and mark it at the top edge instead, labelled with its real value,
    # so it is visible as the outlier it is without hiding the other sixteen.
    bulk = [o for m, o in zip(stats["shared"], stats["M"]) if abs(o) < 15]
    hi = max(bulk) + 1.4
    lo = min(bulk) - 0.8
    for m, s, o in zip(stats["shared"], stats["S"], stats["M"]):
        bij = m in BIJECTIVE
        off = o > hi
        ax.scatter(s, hi if off else o, s=46, zorder=3,
                   marker="^" if off else "o",
                   facecolor="#2b6cb0" if bij else "#c0392b",
                   edgecolor="white", linewidth=0.8)
        ax.annotate(f"{m}  ({o:+.1f}, off scale)" if off else m,
                    (s, hi if off else o), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points", color="0.3")
    ax.set_ylim(lo, hi + 0.9)
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.axvline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_xlabel("single-objective gain  ($\\log_{10}$ regret)", fontsize=9)
    ax.set_ylabel("multi-objective gain  (% hypervolume)", fontsize=9)
    ax.set_title(f"No relationship between the two settings\n"
                 f"Pearson $r={stats['r']:+.2f}$, Spearman "
                 f"$\\rho={stats['rho']:+.2f}$  "
                 f"(without root5: ${stats['r_no5']:+.2f}$, "
                 f"${stats['rho_no5']:+.2f}$)", fontsize=10)
    ax.scatter([], [], facecolor="#2b6cb0", edgecolor="white",
               label="bijective", s=46)
    ax.scatter([], [], facecolor="#c0392b", edgecolor="white",
               label="not bijective", s=46)
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {path}.pdf / .png")


def write_hv_curves(traces, mo, path, inj_cols=2, non_cols=5):
    """Hypervolume against evaluation, bijective block left of a vertical rule.

    Each run is divided by its own hypervolume at the end of the initial
    design.  Raw hypervolume runs from 15 to 129 across the four function
    draws, so pooling them gave interquartile bands spanning the whole panel
    -- all of it draw-to-draw scale, none of it the arms differing.  Both
    arms share the initial design within a run, and therefore share that
    value exactly, so dividing by it removes the scale without favouring
    either and starts every curve at 1.

    Even then the arms sit close: the gains are percent-level, against orders
    of magnitude in the single-objective regret figure.  That is the finding
    rather than a defect of the plot -- only 5 of 12 non-bijective maps reach
    significance -- so each panel prints its gain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    order = sorted(traces, key=lambda m: (m not in BIJECTIVE, m))
    inj = [m for m in order if m in BIJECTIVE]
    non = [m for m in order if m not in BIJECTIVE]
    nrows = max(-(-len(inj) // inj_cols), -(-len(non) // non_cols))
    ncols = inj_cols + non_cols

    fig = plt.figure(figsize=(2.45 * ncols, 2.15 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.46, hspace=0.62)
    grey, blue = "0.45", "#2b6cb0"

    def med_iqr(rows):
        T = list(zip(*rows))
        q = lambda v, f: sorted(v)[max(0, min(len(v) - 1, int(f * len(v))))]
        return ([st.median(c) for c in T],
                [q(c, 0.25) for c in T], [q(c, 0.75) for c in T])

    for block, cols, c0 in ((inj, inj_cols, 0), (non, non_cols, inj_cols)):
        for i, m in enumerate(block):
            ax = fig.add_subplot(gs[i // cols, c0 + i % cols])
            dtr, ctr = traces[m]
            x = range(len(dtr[0]))
            norm = lambda rows: [[v / r[0] for v in r] for r in rows]
            for rows, col, lab in ((norm(dtr), grey, "direct"),
                                   (norm(ctr), blue, "composite")):
                md, lo, hi = med_iqr(rows)
                ax.plot(x, md, color=col, lw=1.4, label=lab, zorder=3)
                ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0)
            gain, _, p = paired_t(mo[m])
            ax.set_title(m, fontsize=9.5)
            ax.text(0.97, 0.06, f"{gain:+.2f}%" + ("*" if p < 0.05 else ""),
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color=blue if p < 0.05 else "0.5",
                    fontweight="bold" if p < 0.05 else "normal")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
            if c0 == 0 and i % cols == 0:
                ax.set_ylabel("HV / HV at $t=0$", fontsize=8)
            if i == 0 and c0 == 0:
                ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    # The rule, drawn in figure coordinates between the two blocks.
    xr = (gs[0, inj_cols].get_position(fig).x0
          + gs[0, inj_cols - 1].get_position(fig).x1) / 2
    fig.add_artist(plt.Line2D([xr, xr], [0.02, 0.93], color="black", lw=1.6))
    fig.text(0.5, 0.985, "Median hypervolume with the interquartile band, "
             "each run relative to its own initial design; higher is better",
             ha="center", fontsize=11)
    fig.text(xr - 0.012, 0.955, "BIJECTIVE", ha="right", fontsize=10.5,
             fontweight="bold", color=blue)
    fig.text(xr + 0.012, 0.955, "NOT BIJECTIVE", ha="left", fontsize=10.5,
             fontweight="bold", color="#8b1a1a")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {path}.pdf / .png ({len(inj)} bijective, "
          f"{len(non)} non-bijective panels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mo", default=os.path.join(
        "..", "MOBO-HighDim", "composite_ablation", "results", "two_stage_matched"))
    ap.add_argument("--so", default="three_arm/results_matern2")
    ap.add_argument("--out", default="note")
    ap.add_argument("--fig", default="figures/so_vs_mo")
    ap.add_argument("--hv-fig", default="figures/hv_curves")
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    if not os.path.isdir(a.mo):
        sys.exit(f"two-objective results not found at {a.mo}; pass --mo")
    mo, so = load_mo(a.mo), load_so(a.so)
    print(f"{len(mo)} multi-objective maps, {len(so)} single-objective maps\n")
    write_mo_table(mo, a.out)
    stats = write_matched_table(so, mo, a.out)
    write_figure(stats, a.fig)
    write_hv_curves(load_traces(a.mo), mo, a.hv_fig)
    print(f"\nmatched on {stats['n']} maps: Pearson {stats['r']:+.3f}, "
          f"Spearman {stats['rho']:+.3f}")
    print(f"   without root5: {stats['r_no5']:+.3f}, {stats['rho_no5']:+.3f}")


if __name__ == "__main__":
    main()
