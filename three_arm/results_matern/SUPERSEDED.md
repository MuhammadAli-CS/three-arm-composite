# Superseded — do not analyse these

This grid was produced before three defects were fixed. Use
`three_arm/results_matern2` instead.

1. **The Matern draws were not reproducible.** `_spectral_sample` scaled the
   frequencies by a Gamma variate drawn with
   `torch.distributions.Gamma(...).sample()`, which takes no generator and
   reads the global RNG. Every construction of a Matern problem was therefore
   a different random function: the draw seed selected nothing, maps did not
   share an `h` (which the design depends on, so that switching `g` cannot
   change the underlying function), and the stored `f_min` cannot be
   recomputed.

2. **`f_min` was not a lower bound where `g` is flat or kinked.** It was found
   by gradient descent on `f`, which cannot move through `clip`'s or `relu`'s
   zero-gradient region. `clip_c2` was beaten by a BO run by 1.4e-2, giving a
   negative regret, and `abs` was negative on three of four draws.

3. **The regret floor of 1e-8 was binding** on 13 of 40 `shiftsq_c0` runs and
   11 of 40 `root4` runs, compressing the very gaps being measured.

Within a single file the comparison is still internally valid — both arms saw
the same `h` and the same `f_min` — so the per-map gains are real
measurements of *some* Matern function. What is not valid is comparing across
maps, since each one saw different functions, and anything involving `abs` or
`clip_c2`.

Kept only so the superseded numbers quoted in older drafts can be traced.
