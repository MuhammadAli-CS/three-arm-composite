"""Run the three-arm experiment on the synthetic test functions.

Single-objective BO with the same LogEI acquisition, candidate pool and paired
initial designs for every arm, so the only thing that differs is the surrogate.

    python three_arm/run.py --check                       # validate arm 3 first
    python three_arm/run.py --group validate --seeds 3 --budget 15
    python three_arm/run.py --group sweep --seeds 10
    python three_arm/run.py --maps shiftsq_c0 abs norm --seeds 10

Reported per condition (paired over seeds, on final log10 regret):
    know g     = direct    - latent      (> 0: knowing g helps)
    observe h  = latent    - composite   (> 0: observing h helps)
    total      = direct    - composite   (the Composite gain in the paper)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from three_arm.arms import ARMS, CompositeArm, DirectArm, LatentArm  # noqa: E402
from three_arm.test_functions import DTYPE, GROUPS, OUTER_MAPS, make_problem  # noqa: E402

warnings.filterwarnings("ignore")


@dataclass
class Config:
    d: int = 6
    lengthscale: float = 0.5
    draw_seed: int = 0
    n0: int = 5
    budget: int = 40
    seeds: int = 10
    n_mc: int = 256
    n_pool: int = 2048
    n_refine: int = 4
    refine_steps: int = 30
    tau: float = 1e-3
    noise: float = 1e-2
    chains: int = 2
    burn_first: int = 500
    burn: int = 150
    keep: int = 200
    thin: int = 2
    shortcuts: bool = True


# --------------------------------------------------------------------------
# Acquisition (identical for every arm)
# --------------------------------------------------------------------------
def log_ei(samples: torch.Tensor, best: float, tau: float) -> torch.Tensor:
    """Smoothed Monte Carlo log expected improvement for minimization.
    samples: (N, m) draws of the objective.  Returns (m,)."""
    z = (best - samples) / tau
    log_sp = torch.where(z > -30.0, torch.nn.functional.softplus(z).clamp_min(1e-300).log(), z)
    return torch.logsumexp(log_sp, dim=0) - math.log(samples.shape[0]) + math.log(tau)


def optimize_acq(arm, best: float, d: int, seed: int, cfg: Config) -> torch.Tensor:
    pool = SobolEngine(d, scramble=True, seed=seed).draw(cfg.n_pool).to(DTYPE)
    with torch.no_grad():
        vals = torch.cat([log_ei(arm.sample_f(pool[i:i + 512]), best, cfg.tau)
                          for i in range(0, len(pool), 512)])
    best_x, best_v = pool[vals.argmax()].clone(), vals.max().item()
    for x0 in pool[vals.argsort(descending=True)[: cfg.n_refine]]:
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.02)
        for _ in range(cfg.refine_steps):
            opt.zero_grad()
            loss = -log_ei(arm.sample_f(x.unsqueeze(0)), best, cfg.tau)[0]
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
        with torch.no_grad():
            v = log_ei(arm.sample_f(x.unsqueeze(0)), best, cfg.tau).item()
        if v > best_v:
            best_x, best_v = x.detach().clone(), v
    return best_x.unsqueeze(0)


# --------------------------------------------------------------------------
# BO loop
# --------------------------------------------------------------------------
def make_arm(name: str, problem, cfg: Config, seed: int):
    if name == "latent":
        return LatentArm(problem, noise=cfg.noise, n_chains=cfg.chains,
                         burn_first=cfg.burn_first, burn=cfg.burn, keep=cfg.keep,
                         thin=cfg.thin, shortcuts=cfg.shortcuts, seed=seed)
    return ARMS[name](problem)


def run_bo(problem, arm_name: str, seed: int, cfg: Config) -> dict:
    X = SobolEngine(problem.d, scramble=True, seed=seed).draw(cfg.n0).to(DTYPE)
    H = problem.h(X)
    y = problem.g(H)
    arm = make_arm(arm_name, problem, cfg, seed)
    best, u_lost, fit_s, acq_s = [y.min().item()], [], 0.0, 0.0
    for t in range(cfg.budget):
        t0 = time.time()
        arm.fit(X, H, y)
        t1 = time.time()
        arm.prepare_samples(cfg.n_mc, seed * 7919 + t)
        x = optimize_acq(arm, y.min().item(), problem.d, seed * 104729 + t, cfg)  # pool shared across arms
        fit_s, acq_s = fit_s + (t1 - t0), acq_s + (time.time() - t1)
        h = problem.h(x)
        X, H, y = torch.cat([X, x]), torch.cat([H, h]), torch.cat([y, problem.g(h)])
        best.append(y.min().item())
        u_lost.append(arm.diagnostics().get("u_lost", float("nan")))
    return {"best": best, "u_lost": u_lost, "fit_seconds": fit_s, "acq_seconds": acq_s,
            "final_stats": arm.diagnostics()}


def log_regret(value: float, f_min: float) -> float:
    return math.log10(max(value - f_min, 1e-8))


def paired(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) if n > 1 else float("nan")
    return mean, se


def run_condition(map_name: str, arms, cfg: Config, out_dir: Path, quiet: bool = False) -> dict:
    problem = make_problem(map_name, d=cfg.d, lengthscale=cfg.lengthscale, draw_seed=cfg.draw_seed)
    if not quiet:
        print(f"\n== {problem.describe()}   (f_min ~ {problem.f_min:.4f})")
    results = {a: [] for a in arms}
    for seed in range(cfg.seeds):
        for a in arms:
            r = run_bo(problem, a, seed, cfg)
            results[a].append(r)
            if not quiet:
                print(f"   seed {seed:2d}  {a:9s}  final log10 regret "
                      f"{log_regret(r['best'][-1], problem.f_min):7.3f}"
                      f"   ({r['fit_seconds'] + r['acq_seconds']:.1f}s)", flush=True)

    final = {a: [log_regret(r["best"][-1], problem.f_min) for r in results[a]] for a in arms}
    summary = {a: sum(v) / len(v) for a, v in final.items()}
    if {"direct", "latent"} <= set(arms):
        summary["know_g"] = paired(final["direct"], final["latent"])
    if {"latent", "composite"} <= set(arms):
        summary["observe_h"] = paired(final["latent"], final["composite"])
    if {"direct", "composite"} <= set(arms):
        summary["total"] = paired(final["direct"], final["composite"])
    summary["n"], summary["draws"] = cfg.seeds, 1
    if "latent" in arms:
        summary["mean_u_lost"] = sum(sum(r["u_lost"]) / len(r["u_lost"]) for r in results["latent"]) / cfg.seeds

    out_dir.mkdir(parents=True, exist_ok=True)
    # One file per (map, function draw).  Draw 0 keeps the bare name so older
    # single-draw result directories still collect.
    stem = map_name if cfg.draw_seed == 0 else f"{map_name}@d{cfg.draw_seed}"
    with open(out_dir / f"{stem}.json", "w") as fh:
        json.dump({"map": map_name, "config": asdict(cfg), "f_min": problem.f_min,
                   "describe": problem.describe(), "summary": summary, "results": results}, fh, indent=1)
    return summary


def _worker(map_name: str, arms, cfg: Config, out: str) -> dict:
    """Entry point for one parallel process: one map, all seeds, all arms."""
    torch.set_num_threads(1)
    torch.set_default_dtype(DTYPE)
    warnings.filterwarnings("ignore")
    return run_condition(map_name, arms, cfg, Path(out), quiet=True)


def print_table(rows):
    print("\n" + "=" * 110)
    print(f"{'map':14s} {'direct':>8s} {'comp':>8s} {'latent':>8s}   {'know g':>15s} {'observe h':>15s} {'U_lost':>8s} "
          f"{'runs':>5s} {'drw':>4s}")
    print("-" * 110)
    for name, s in rows:
        fmt = lambda key: (f"{s[key][0]:+6.3f}+/-{s[key][1]:5.3f}" if key in s else "      -      ")
        print(f"{name:14s} {s.get('direct', float('nan')):8.3f} {s.get('composite', float('nan')):8.3f} "
              f"{s.get('latent', float('nan')):8.3f}   {fmt('know_g'):>15s} {fmt('observe_h'):>15s} "
              f"{s.get('mean_u_lost', float('nan')):8.4f} {s.get('n', 0):5d} {s.get('draws', 1):4d}")
    print("=" * 110)
    print("final log10 regret (lower is better); differences are paired over seeds, > 0 favours the second arm")


# --------------------------------------------------------------------------
# Sampler validation
# --------------------------------------------------------------------------
def _mixing(arm: LatentArm, Xt: torch.Tensor):
    """Across-chain diagnostics on the predictive of f at held-out points --
    invariant to symmetries of h that leave f unchanged.

    Returns (R-hat, disagreement): Gelman-Rubin R-hat on per-sample predictive
    means, and the largest spread between chains' predictive means in objective
    units.  R-hat is unreliable when the posterior is nearly a point (tiny
    within-chain variance); the disagreement is what matters for BO."""
    if len(arm.chain_F) < 2:
        return float("nan"), float("nan")
    eps = torch.randn(16, arm.prior.k, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    per_chain = []
    for F in arm.chain_F:
        mean, sd = arm.prior.conditional(arm.X, F, Xt, arm.L)
        per_chain.append(arm.problem.g(mean.unsqueeze(0) + sd * eps[:, None, None, :]).mean(0))  # (S, m)
    P = torch.stack(per_chain)                                  # (C, S, m)
    S = P.shape[1]
    W = P.var(1).mean(0)
    B = S * P.mean(1).var(0)
    rhat = (((S - 1) / S * W + B / S) / W.clamp_min(1e-12)).sqrt()
    chain_means = P.mean(1)
    disagreement = float((chain_means.max(0).values - chain_means.min(0).values).max())
    return float(rhat.max()), disagreement


def run_checks(cfg: Config) -> None:
    n_train = 20
    X = SobolEngine(cfg.d, scramble=True, seed=123).draw(n_train).to(DTYPE)
    Xt = SobolEngine(cfg.d, scramble=True, seed=456).draw(64).to(DTYPE)
    mk = lambda p, seed: LatentArm(p, noise=cfg.noise, n_chains=4, burn_first=800, keep=1500,
                                   thin=5, shortcuts=False, seed=seed)

    print("[1] linear map (sum): sampler vs exact Gaussian posterior")
    p = make_problem("sum", d=cfg.d, lengthscale=cfg.lengthscale, draw_seed=cfg.draw_seed)
    y = p.f(X)
    arm = mk(p, 1)
    t0 = time.time()
    arm.fit(X, None, y)
    ex_mean, ex_cov = arm.exact_linear_posterior(X, y)
    ex_sd = ex_cov.diagonal().clamp_min(1e-12).sqrt().reshape(p.outer.k, n_train).T
    z = ((arm.F.mean(0) - ex_mean).abs() / ex_sd).flatten()
    ratio = (arm.F.std(0) / ex_sd).flatten()
    ok1 = float(z.median()) < 0.3 and 0.6 < float(ratio.median()) < 1.6
    print(f"    mean error / exact sd: median {z.median():.3f}, max {z.max():.3f}")
    print(f"    sd ratio (sampler / exact): median {ratio.median():.3f}, range [{ratio.min():.2f}, {ratio.max():.2f}]")
    rh, dis = _mixing(arm, Xt)
    print(f"    predictive R-hat {rh:.3f}, chain disagreement {dis:.4f}   move accept {arm.stats['move_accept']:.2f}"
          f"   ({time.time() - t0:.1f}s)   -> {'PASS' if ok1 else 'CHECK'}")

    print("[2] invertible map (exp_b1): true h should sit inside the sampler's posterior")
    p = make_problem("exp_b1", d=cfg.d, lengthscale=cfg.lengthscale, draw_seed=cfg.draw_seed)
    y = p.f(X)
    arm = mk(p, 2)
    arm.fit(X, None, y)
    h_true = p.h(X)
    err = (arm.F.mean(0) - h_true).abs().flatten()
    z = err / arm.F.std(0).clamp_min(1e-3).flatten()
    cover = float((z < 2.0).double().mean())
    ok2 = cover >= 0.8 and float(z.max()) < 4.0
    worst = int(err.argmax())
    print(f"    |posterior mean - true h| / posterior sd: median {z.median():.2f}, max {z.max():.2f}; "
          f"within 2 sd: {cover:.0%}")
    print(f"    largest raw error {err.max():.3f} at true h = {h_true.flatten()[worst]:+.2f} "
          f"(exp is flat there, so h is only weakly identified at noise {cfg.noise:g})")
    print(f"    U_lost = {arm.stats['u_lost']:.2e}   -> {'PASS' if ok2 else 'CHECK'}")

    print("[3] information-loss sweep: U_lost should fall as c grows")
    print(f"    {'map':14s} {'U_lost':>9s} {'R-hat':>7s} {'disagree':>9s} {'move acc':>9s}")
    u_vals = []
    for name in GROUPS["sweep"] + ["abs"]:
        p = make_problem(name, d=cfg.d, lengthscale=cfg.lengthscale, draw_seed=cfg.draw_seed)
        arm = mk(p, 3)
        arm.fit(X, None, p.f(X))
        u_vals.append(arm.stats["u_lost"])
        rh, dis = _mixing(arm, Xt)
        print(f"    {name:14s} {arm.stats['u_lost']:9.4f} {rh:7.3f} {dis:9.4f} {arm.stats['move_accept']:9.2f}")
    sweep_u = u_vals[: len(GROUPS["sweep"])]
    ok3 = sweep_u[0] > sweep_u[-1]
    print(f"    U_lost(c=0) > U_lost(c=3): {'PASS' if ok3 else 'CHECK'}")
    print("\nDiagnostics are on the predictive of f, so sign symmetries in h do not inflate them.\n"
          "Chain disagreement is in normalized objective units (the range is ~1); under ~0.02 is fine.\n"
          "R-hat is unreliable when U_lost is near zero -- the posterior is then almost a point.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate the latent-arm sampler and exit")
    ap.add_argument("--group", choices=sorted(GROUPS), help="named set of outer maps")
    ap.add_argument("--maps", nargs="+", choices=sorted(OUTER_MAPS), help="explicit outer maps")
    ap.add_argument("--arms", nargs="+", default=["direct", "composite", "latent"], choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, default=Config.seeds)
    ap.add_argument("--budget", type=int, default=Config.budget)
    ap.add_argument("--n0", type=int, default=Config.n0)
    ap.add_argument("--d", type=int, default=Config.d)
    ap.add_argument("--noise", type=float, default=Config.noise)
    ap.add_argument("--draw-seed", type=int, default=Config.draw_seed,
                    help="which random function h to draw; vary it so conclusions "
                         "are not a property of one sampled function")
    ap.add_argument("--no-shortcuts", action="store_true",
                    help="sample arm 3 even for invertible / linear maps")
    ap.add_argument("--collect", action="store_true",
                    help="rebuild the summary table from saved JSON (use after cluster jobs land)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many maps in parallel, one thread each (try 16 on a 20-core box)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    cfg = Config(d=args.d, n0=args.n0, budget=args.budget, seeds=args.seeds,
                 noise=args.noise, shortcuts=not args.no_shortcuts,
                 draw_seed=args.draw_seed)
    torch.set_default_dtype(DTYPE)

    if args.check:
        run_checks(cfg)
        return

    if args.collect:
        order = list(OUTER_MAPS)
        files = sorted(Path(args.out).glob("*.json"))
        if not files:
            ap.error(f"no results in {args.out}")
        # Pool the function draws of one map: each file contributes its own
        # per-seed final regrets, and the paired statistics are recomputed over
        # the pooled (draw, seed) pairs.  Pairing stays valid because every arm
        # within a file saw the same draw and the same initial designs.
        pooled, draws = {}, {}
        for path in files:
            blob = json.loads(path.read_text())
            name = blob["map"]
            acc = pooled.setdefault(name, {})
            draws[name] = draws.get(name, 0) + 1
            for arm, runs in blob["results"].items():
                acc.setdefault(arm, []).extend(
                    log_regret(r["best"][-1], blob["f_min"]) for r in runs)
            if "latent" in blob["results"]:
                acc.setdefault("_u", []).extend(
                    sum(r["u_lost"]) / len(r["u_lost"]) for r in blob["results"]["latent"])
        rows = []
        for name in sorted(pooled, key=lambda n: order.index(n) if n in order else len(order)):
            acc = pooled[name]
            u = acc.pop("_u", None)
            s_ = {a: sum(v) / len(v) for a, v in acc.items()}
            if {"direct", "latent"} <= set(acc):
                s_["know_g"] = paired(acc["direct"], acc["latent"])
            if {"latent", "composite"} <= set(acc):
                s_["observe_h"] = paired(acc["latent"], acc["composite"])
            if {"direct", "composite"} <= set(acc):
                s_["total"] = paired(acc["direct"], acc["composite"])
            if u:
                s_["mean_u_lost"] = sum(u) / len(u)
            s_["n"], s_["draws"] = len(next(iter(acc.values()))), draws[name]
            rows.append((name, s_))
        print_table(rows)
        return

    names = list(dict.fromkeys((GROUPS[args.group] if args.group else []) + (args.maps or [])))
    if not names:
        ap.error("give --group, --maps or --check")

    if args.jobs > 1 and len(names) > 1:
        # One process per map.  The matrices here are tiny (n <= 45), so BLAS
        # threading does not help; one thread per worker and many workers does.
        import concurrent.futures as cf
        rows, done = [], 0
        with cf.ProcessPoolExecutor(max_workers=min(args.jobs, len(names))) as ex:
            futures = {ex.submit(_worker, n, args.arms, cfg, args.out): n for n in names}
            for fut in cf.as_completed(futures):
                name, summary = futures[fut], fut.result()
                rows.append((name, summary))
                done += 1
                print(f"[{done}/{len(names)}] {name} done", flush=True)
        rows.sort(key=lambda r: names.index(r[0]))
    else:
        rows = [(n, run_condition(n, args.arms, cfg, Path(args.out))) for n in names]
    print_table(rows)


if __name__ == "__main__":
    main()
