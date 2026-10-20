"""Predictive accuracy of Composite vs Direct, with no optimizer in the loop.

The posterior-equivalence proposition is a statement about conditional
distributions, and optimization regret is a bad instrument for it: if g is
injective it is monotone, so its optimum over the reachable range of h is
always at an endpoint, while a non-injective g can have an interior one.
Injectivity therefore decides what the optimization problem *is*, and any
regret comparison between the two classes is confounded by construction.

Predictive accuracy at held-out points has no such problem.  Both models are
fitted to the same n observations and scored on the same held-out f values:

    CRPS   proper scoring rule, computed exactly from samples, so it grades the
           whole predictive distribution rather than just its centre
    RMSE   error of the predictive mean alone
    KS     Kolmogorov distance between the two predictive CDFs -- how different
           the models are, regardless of which is better

    python three_arm/predictive.py --group bijectivity
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import torch
from torch.quasirandom import SobolEngine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.arms import CompositeArm, DirectArm                     # noqa: E402
from three_arm.run import paired                                       # noqa: E402
from three_arm.test_functions import DTYPE, GROUPS, OUTER_MAPS, make_problem  # noqa: E402

warnings.filterwarnings("ignore")
N_SAMPLES = 512


def crps(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """CRPS from samples: E|X - y| - 0.5 E|X - X'|, lower is better.

    Exact for a sample-based predictive, which the composite arm's is, so no
    density estimation is needed anywhere.  Both terms use the same draws.
    """
    s = samples                                           # (S, m)
    term1 = (s - truth.unsqueeze(0)).abs().mean(0)
    k = min(256, s.shape[0])                              # second term is O(k^2)
    a = s[:k].unsqueeze(1)                                # (k,1,m)
    b = s[:k].unsqueeze(0)                                # (1,k,m)
    term2 = 0.5 * (a - b).abs().mean(dim=(0, 1))
    return term1 - term2


def ks_distance(sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """Max gap between two sample CDFs, evaluated on the pooled sample points."""
    pooled = torch.cat([sa, sb], 0)                                   # (2S, m)
    fa = (sa.unsqueeze(1) <= pooled.unsqueeze(0)).to(DTYPE).mean(0)   # (2S, m)
    fb = (sb.unsqueeze(1) <= pooled.unsqueeze(0)).to(DTYPE).mean(0)
    return (fa - fb).abs().max(0).values


def one_fit(problem, n: int, seed: int, Xt: torch.Tensor, ft: torch.Tensor):
    X = SobolEngine(problem.d, scramble=True, seed=seed).draw(n).to(DTYPE)
    H, y = problem.h(X), problem.f(X)

    direct = DirectArm(problem); direct.fit(X, H, y)
    comp = CompositeArm(problem); comp.fit(X, H, y)
    for arm in (direct, comp):
        arm.prepare_samples(N_SAMPLES, seed * 31 + 7)
    sd_, sc_ = direct.sample_f(Xt), comp.sample_f(Xt)

    return dict(
        crps_direct=float(crps(sd_, ft).mean()),
        crps_comp=float(crps(sc_, ft).mean()),
        rmse_direct=float(((sd_.mean(0) - ft) ** 2).mean().sqrt()),
        rmse_comp=float(((sc_.mean(0) - ft) ** 2).mean().sqrt()),
        ks=float(ks_distance(sd_, sc_).mean()),
    )


def run_one_draw(args):
    """One (map, draw) unit.  Parallelising here rather than over maps keeps
    every worker busy when there are fewer maps than cores."""
    name, draw, ns, reps, d, n_test = args
    torch.set_num_threads(1)
    torch.set_default_dtype(DTYPE)
    warnings.filterwarnings("ignore")
    p = make_problem(name, d=d, lengthscale=0.5, draw_seed=draw)
    Xt = SobolEngine(d, scramble=True, seed=9001 + draw).draw(n_test).to(DTYPE)
    ft = p.f(Xt)
    out = {n: {k: [] for k in ("cd", "cc", "rd", "rc", "ks")} for n in ns}
    for n in ns:
        for r in range(reps):
            res = one_fit(p, n, 1000 * draw + r, Xt, ft)
            out[n]["cd"].append(res["crps_direct"])
            out[n]["cc"].append(res["crps_comp"])
            out[n]["rd"].append(res["rmse_direct"])
            out[n]["rc"].append(res["rmse_comp"])
            out[n]["ks"].append(res["ks"])
    return name, out


def _merge(dest, src, ns):
    for n in ns:
        for k in src[n]:
            dest[n][k].extend(src[n][k])


def run_map(name, ns, draws, reps, d=6, n_test=512):
    out = {n: {k: [] for k in ("cd", "cc", "rd", "rc", "ks")} for n in ns}
    for draw in range(draws):
        _, part = run_one_draw((name, draw, ns, reps, d, n_test))
        _merge(out, part, ns)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="bijectivity", choices=sorted(GROUPS))
    ap.add_argument("--maps", nargs="+")
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="write the raw per-fit records here as JSON")
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    names = a.maps or GROUPS[a.group]
    from three_arm.injectivity import summarize
    nbar = {m: summarize(m)["nbar"] for m in names}

    if a.jobs > 1:
        import concurrent.futures as cf
        units = [(m, d_, a.ns, a.reps, 6, 512) for m in names for d_ in range(a.draws)]
        results = {m: {n: {k: [] for k in ("cd", "cc", "rd", "rc", "ks")}
                       for n in a.ns} for m in names}
        done = 0
        # Workers are recycled every few units: a long-lived worker grows
        # to well over 2 GB across repeated GP fits, and sixteen of those
        # exhaust memory and break the pool.
        with cf.ProcessPoolExecutor(max_workers=a.jobs,
                                    max_tasks_per_child=6) as ex:
            for name, part in ex.map(run_one_draw, units):
                _merge(results[name], part, a.ns)
                done += 1
                print(f"  {done}/{len(units)} units", end="\r", flush=True)
        print(" " * 30, end="\r")
    else:
        results = {m: run_map(m, a.ns, a.draws, a.reps) for m in names}

    if a.out:
        import json
        blob = {"ns": a.ns, "draws": a.draws, "reps": a.reps, "nbar": nbar,
                "kind": {m: OUTER_MAPS[m].kind for m in names},
                "results": {m: {str(n): results[m][n] for n in a.ns} for m in names}}
        with open(a.out, "w") as fh:
            json.dump(blob, fh)
        print(f"wrote {a.out}")

    reps_total = a.draws * a.reps
    for n in a.ns:
        print(f"\n=== n = {n} training points, {reps_total} fits per map "
              f"({a.draws} draws x {a.reps} designs) ===")
        print(f"{'map':9s} {'kind':18s} {'nbar':>5s} {'CRPS gain':>16s} "
              f"{'RMSE gain':>16s} {'KS':>7s} {'wins':>7s}")
        print("-" * 86)
        for m in names:
            o = results[m][n]
            g, se = paired(o["cd"], o["cc"])          # positive favours composite
            rg, rse = paired(o["rd"], o["rc"])
            w = sum(1 for x, y in zip(o["cd"], o["cc"]) if y < x)
            kind = OUTER_MAPS[m].kind
            print(f"{m:9s} {kind:18s} {nbar[m]:5.2f} {g:+8.4f} +/- {se:5.4f} "
                  f"{rg:+8.4f} +/- {rse:5.4f} {sum(o['ks'])/len(o['ks']):7.3f} "
                  f"{w:4d}/{reps_total}")
    print("\nCRPS and RMSE gains are direct - composite, so positive favours composite.")
    print("KS is the mean Kolmogorov distance between the two predictive CDFs.")


if __name__ == "__main__":
    main()
