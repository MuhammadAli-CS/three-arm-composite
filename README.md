# Three-arm composite experiment: knowing *g* versus observing *h*

Composite Bayesian optimization on `f(x) = g(h(x))` gives a surrogate two
advantages at once:

1. **knowledge of `g`** — it is handed the outer function and never learns it from data;
2. **observation of `h`** — each expensive evaluation returns `k` numbers instead of one.

Every direct-vs-composite comparison changes both together, so none of them can
say which advantage produced the gain. This repo adds a third surrogate that
receives exactly one of the two, which splits the composite gain into its parts.

| arm | observes | knows `g` | surrogate |
|---|---|---|---|
| 1. direct | `y = f(x)` | no | GP on `y`, hyperparameters by MLE |
| 2. composite | `h(x)` | yes | GPs on each `h_j`, then apply `g` |
| 3. **latent** | `y = f(x)` | yes | same GP prior on `h`, `h` inferred from `y` by MCMC |

```
latent    - direct    = the value of knowing g
composite - latent    = the value of observing h
```

and the two sum to the composite gain by construction.

**[`slides/slides.pdf`](slides/slides.pdf)** is a 16-slide meeting deck.
**[`three_arm/design.pdf`](three_arm/design.pdf) is the full write-up** — a short
overview first, then the reasoning, the test problems, the inference procedure,
every measured quantity, and the validation evidence.

## Status

The harness is validated and the main grid has run. **The headline numbers are
not yet trustworthy** — see [Open problems](#open-problems). Two results do hold:

- **The invertible controls are exact.** `observe h = -0.000 ± 0.000` on
  `identity`, `exp_b1`, `sinh_b1`, where theory says it must be zero.
- **Censoring away from the optimum costs nothing; censoring at the optimum
  costs a lot.** `relu` (optimum where `g` is injective) gives `observe h =
  +0.343 ± 0.209`; `clip_c2` (censored coordinate matters at the optimum) gives
  `+1.517 ± 0.421`. This was predicted in advance and is the design's one
  non-obvious claim.

## Quick start

```bash
pip install -r requirements.txt

python three_arm/run.py --check                    # validate the sampler first
python three_arm/run.py --group sweep --seeds 5 --jobs 16
python three_arm/run.py --collect --out three_arm/results
```

Reproduce the stored grid (19 maps × 4 function draws × 5 seeds, ~40 min on 16 cores):

```bash
MAPS="identity sum exp_b1 sinh_b1 shiftsq_c0 shiftsq_c0.5 shiftsq_c1 shiftsq_c2 shiftsq_c3       abs cos_w1 cos_w2 cos_w4 relu clip_c2 product norm max ratio"
for d in 0 1 2 3; do
  python three_arm/run.py --maps $MAPS --seeds 5 --draw-seed $d --jobs 16 --out results/main
done
python three_arm/run.py --collect --out results/main
```

On SLURM:

```bash
DRAWS="0 1 2 3" SEEDS=5 bash cluster/submit_three_arm.sh
```

## Layout

```
three_arm/test_functions.py   RFF function draws, the 23 outer maps, level-set moves
three_arm/arms.py             the three surrogates; LatentArm holds the sampler
three_arm/run.py              BO loop, acquisition, validation checks, collection
three_arm/design.tex/.pdf     the design document
slides/slides.tex/.pdf        meeting deck
cluster/                      SLURM submission (one job per map × draw)
results/main/                 19 maps × 4 draws × 5 seeds
results/identity_no_shortcut/ the control that forces MCMC onto a known answer
```

Read results with `python three_arm/run.py --collect --out results/main`, which
pools the function draws and recomputes the paired statistics over the pooled
(draw, seed) pairs.

## Validation

Three checks, each on a condition whose answer is known independently of the
code being graded. `python three_arm/run.py --check` runs all three in ~1 min.

| check | what it grades | result |
|---|---|---|
| linear map vs. exact Gaussian posterior | sampler correctness | mean error 0.069 posterior sd; sd ratio 0.968 |
| injective map vs. known `h` | calibration | 100% within 2 sd |
| information-loss sweep | the designed knob | `U_lost` falls 0.870 → 0.004 |

These exist because they work. An earlier version of the grid looked entirely
reasonable — smooth trends, sensible signs, tight error bars — and was wrong;
what exposed it was `identity`, where `observe h` must be exactly 0 and the code
reported +0.956. Three defects were found and fixed (design doc §Defects).

## Open problems

**1. `observe h` carries an additive bias of unknown size.** `identity` run with
`--no-shortcuts`, forcing the general sampler onto a problem where `h` is fully
recoverable, reports `observe h = +0.409 ± 0.133` when the truth is 0 — while
`U_lost = 0.0016`, so the sampler *is* recovering `h`. The gap is therefore not
information loss. The likely cause is Monte Carlo: composite draws `n_mc`
samples from one exact Gaussian conditional, while latent spreads the same
budget across a mixture over latent samples. A first test at `n_mc = 4096` drove
the gap to `-0.008 ± 0.357`, but composite simultaneously moved 0.70 the wrong
way, which extra samples cannot cause — so that run is seed noise and the
question is unresolved.

**2. The sweep runs backwards.** `observe h` should fall toward 0 as `c` grows
and `g` becomes injective. It rises: +1.899, +1.688, +2.303, +3.159, +2.923. The
anomaly survives subtracting the bias in (1). Ruled out as causes: sampler
budget (heavy settings change `U_lost` by nothing) and warm starts (cold fits on
the identical design differ in no consistent direction).

**3. Level-set moves are dead on realistic designs.** Acceptance is 0.006–0.074
in the BO loop and exactly 0.000 on clustered designs, against 0.5–0.68 on the
spread designs `--check` uses: single-site branch flips are almost always
rejected once neighbouring points are strongly coupled under the prior. Block
moves — flipping all points within a lengthscale of a random centre — keep the
prior ratio closed-form and symmetric. This is a real defect, but it does not
explain (1) or (2), since `identity` has no branches at all.

**4. The hyperparameter asymmetry is measured, not removed.** Arms 2 and 3 get
the generator's true kernel; arm 1 fits its own by MLE. `identity` puts that
advantage alone at `+0.409 ± 0.255`, comparable to the real effects, so
`know g` should be read as an upper bound until all three arms run with MLE
hyperparameters.

**5. Nothing here has `k > 2`.** The real benchmarks have 4–8 intermediates.

## Environment

Python 3.11, torch 2.12 (CPU), botorch 0.9.5, gpytorch 1.11. CPU only — the
matrices are at most 45×45, so a GPU buys nothing and BLAS threading measurably
does not help. Parallelism comes from running many single-threaded jobs.
