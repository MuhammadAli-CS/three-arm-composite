# Pre-Matern three-arm runs (RBF, oracle composite)

19 maps x 4 draws x 5 seeds, all three arms (`direct`, `composite`, `latent`),
budget 40. These predate the `kernel` config field, so `config.kernel` is
`null`: `h` was drawn from an **RBF** kernel while both surrogates assumed
BoTorch's default **Matern-5/2**, and `composite` here is the *oracle* arm,
handed the generator's true kernel rather than fitting its own.

Those two advantages are what produced the original "Composite wins on nearly
every problem, bijective or not" claim. See `three_arm/results_matern2` for
the corrected grid and `note/problems.pdf` for the retraction.

Rescued from `MOBO-HighDim/three_arm/`, where they were untracked and would
have been lost when that stale copy of the package was removed. Kept so the
superseded numbers in older drafts can be traced.
