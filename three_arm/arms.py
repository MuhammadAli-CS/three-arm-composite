"""The three surrogate arms, behind one interface.

Every arm is fit to the observed data and then produces Monte Carlo samples of
the normalized objective at candidate points; the run wrapper turns those into
the same LogEI acquisition for all three.

    DirectArm     GP on y, hyperparameters by MLE (as in the paper).
    CompositeArm  known GP prior on each h_j, conditioned on the observed h.
    LatentArm     the same known prior, conditioned only on y = g(h) + noise.

Composite and Latent share the prior exactly -- the generator's true kernel --
so the only difference between them is what they observe.

Interface: ``fit(X, H, y)``, ``prepare_samples(n, seed)``, ``sample_f(Xc)``
returning an (n_samples, m) tensor, and ``diagnostics()``.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double
TWO_PI = 2.0 * math.pi


def rbf(X1: torch.Tensor, X2: torch.Tensor, lengthscale: float) -> torch.Tensor:
    """Unit-outputscale RBF kernel.  Squared distances are formed explicitly so
    gradients stay finite when a candidate coincides with a training point."""
    A, B = X1 / lengthscale, X2 / lengthscale
    d2 = (A * A).sum(-1, keepdim=True) - 2.0 * A @ B.transpose(-1, -2) + (B * B).sum(-1).unsqueeze(-2)
    return torch.exp(-0.5 * d2.clamp_min(0.0))


class KnownPrior:
    """Independent priors h_j ~ GP(m_j, v_j k_RBF(ell)), shared by arms 2 and 3."""

    def __init__(self, means, variances, lengthscale: float, jitter: float = 1e-6):
        self.m = torch.as_tensor(means, dtype=DTYPE)
        self.v = torch.as_tensor(variances, dtype=DTYPE)
        self.ls, self.jitter = float(lengthscale), jitter
        self.k = self.m.numel()

    @classmethod
    def from_problem(cls, problem) -> "KnownPrior":
        return cls(problem.prior_means, problem.prior_vars, problem.lengthscale)

    def unit_chol(self, X: torch.Tensor) -> torch.Tensor:
        K = rbf(X, X, self.ls) + self.jitter * torch.eye(len(X), dtype=DTYPE)
        return torch.linalg.cholesky(K)

    def conditional(self, X, F, Xc, L=None):
        """Condition each h_j on its values F (S, n, k) at X.
        Returns mean (S, m, k) and sd (m, k) at Xc (the sd does not depend on F)."""
        L = self.unit_chol(X) if L is None else L
        Ks = rbf(X, Xc, self.ls)                       # (n, m)
        A = torch.cholesky_solve(Ks, L)                # K^{-1} Ks
        mean = self.m + torch.einsum("nm,snk->smk", A, F - self.m)
        var_unit = (1.0 - (Ks * A).sum(0)).clamp_min(1e-12)
        return mean, (var_unit.unsqueeze(-1) * self.v).sqrt()


# ==========================================================================
# Arm 1: Direct
# ==========================================================================
class DirectArm:
    name = "direct"

    def __init__(self, problem):
        self.problem = problem

    def fit(self, X, H, y):
        del H  # never used by the direct arm
        self.model = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:  # keep the initial hyperparameters if fitting fails
            self.model.eval()

    def prepare_samples(self, n: int, seed: int):
        self.eps = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)

    def sample_f(self, Xc):
        post = self.model.posterior(Xc)
        mean = post.mean.squeeze(-1)
        sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        return mean.unsqueeze(0) + sd.unsqueeze(0) * self.eps.unsqueeze(-1)

    def diagnostics(self):
        return {}


# ==========================================================================
# Arm 2: Composite (observes h)
# ==========================================================================
class CompositeArm:
    name = "composite"

    def __init__(self, problem):
        self.problem = problem
        self.prior = KnownPrior.from_problem(problem)

    def fit(self, X, H, y):
        self.X, self.F, self.L = X, H.unsqueeze(0), self.prior.unit_chol(X)

    def prepare_samples(self, n: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.eps = torch.randn(n, self.prior.k, generator=gen, dtype=DTYPE)

    def sample_f(self, Xc):
        mean, sd = self.prior.conditional(self.X, self.F, Xc, self.L)   # (1,m,k), (m,k)
        return self.problem.g(mean + sd * self.eps[:, None, :])

    def diagnostics(self):
        return {"u_lost": 0.0}


# ==========================================================================
# Arm 3: Latent (observes only y; h is inferred)
# ==========================================================================
class LatentArm:
    """Bayesian inference on g(h) with h ~ known GP and h never observed.

    Posterior over the latent values h(X) at the evaluated inputs:
        p(h(X) | y)  proportional to  prod_i N(y_i; g(h(x_i)), noise^2) * N(h(X); m, v K)
    Prediction at new points is then exact given each sample (a GP
    conditional), so the predictive is a mixture of GPs.

    Sampling uses two kinds of move per sweep:
      * elliptical slice sampling (Murray, Adams & MacKay 2010) on the whole
        latent vector -- gradient-free, so kinks in g are fine;
      * level-set moves supplied by the outer map, which change h(x_i) without
        changing g(h(x_i)) and are accepted on the GP prior ratio alone.  These
        cross exactly the ambiguity g creates (sign flips for z^2, rotations
        for the norm, ...), which ESS alone crosses very slowly when the
        likelihood is tight.
    Burn-in anneals the likelihood noise down to ``noise`` so chains started
    from the prior find the data.  Later fits warm-start from the previous
    chain state.

    Shortcuts (on by default): identity / invertible maps use h = g^{-1}(y)
    exactly, which makes this arm identical to Composite; linear maps use the
    exact Gaussian posterior.  ``run.py --check`` verifies both against the
    sampler.
    """

    name = "latent"

    def __init__(self, problem, noise: float = 1e-2, n_chains: int = 2,
                 burn_first: int = 500, burn: int = 150, keep: int = 200, thin: int = 2,
                 moves_per_sweep: int = 8, shortcuts: bool = True, seed: int = 0):
        self.problem, self.outer = problem, problem.outer
        self.prior = KnownPrior.from_problem(problem)
        self.noise, self.n_chains = noise, n_chains
        self.burn_first, self.burn, self.keep, self.thin = burn_first, burn, keep, thin
        self.moves_per_sweep, self.shortcuts = moves_per_sweep, shortcuts
        self.gen = torch.Generator().manual_seed(seed)
        self.states: Optional[List[torch.Tensor]] = None
        self.stats: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X, H, y):
        del H  # the latent arm never sees the intermediates
        self.X, self.L = X, self.prior.unit_chol(X)
        kind = self.outer.kind
        if self.shortcuts and kind in ("identity", "invertible"):
            self.F = self.problem.h_from_y(y).unsqueeze(0)
            self.chain_F = [self.F]
            self.stats = {"u_lost": 0.0, "method": "inverse"}
            return
        if self.shortcuts and kind == "linear":
            n_samp = self.n_chains * max(1, self.keep // self.thin)
            self.F = self.sample_exact_linear(X, y, n_samp)
            self.chain_F = [self.F]
            self.stats = {"u_lost": self.F.var(0).mean().item(), "method": "exact-linear"}
            return
        self._mcmc(X, y)

    # ------------------------------------------------------ exact linear case
    def exact_linear_posterior(self, X, y):
        """Gaussian posterior of the latents when y = a sum_j c_j h_j(X) + b + noise.
        Returns the mean as (n, k) and the covariance over the stacked (k*n,) vector."""
        n, k = len(X), self.prior.k
        c = torch.as_tensor(self.outer.linear_coef, dtype=DTYPE)
        a = 1.0 / (self.problem.q99 - self.problem.q01)
        b = -self.problem.q01 * a
        K = rbf(X, X, self.prior.ls) + self.prior.jitter * torch.eye(n, dtype=DTYPE)
        Sigma = torch.block_diag(*[self.prior.v[j] * K for j in range(k)])
        mvec = self.prior.m.repeat_interleave(n)
        A = a * torch.cat([c[j] * torch.eye(n, dtype=DTYPE) for j in range(k)], dim=1)
        S = A @ Sigma @ A.T + self.noise ** 2 * torch.eye(n, dtype=DTYPE)
        G = torch.linalg.solve(S, A @ Sigma).T          # Sigma A^T S^{-1}
        mean = mvec + G @ (y - b - A @ mvec)
        cov = Sigma - G @ A @ Sigma
        return mean.reshape(k, n).T, 0.5 * (cov + cov.T)

    def sample_exact_linear(self, X, y, n_samples):
        mean, cov = self.exact_linear_posterior(X, y)
        evals, evecs = torch.linalg.eigh(cov)
        root = evecs * evals.clamp_min(0.0).sqrt()
        z = torch.randn(n_samples, cov.shape[0], generator=self.gen, dtype=DTYPE)
        flat = mean.T.reshape(-1) + z @ root.T                  # (S, k*n)
        n, k = mean.shape
        return flat.reshape(n_samples, k, n).transpose(1, 2)     # (S, n, k)

    # ------------------------------------------------------------- sampler
    def _loglik(self, H, y, sigma):
        r = (y - self.problem.g(H)) / sigma
        return -0.5 * float((r * r).sum())

    def _ess_step(self, H, y, sigma, ll):
        n, k = H.shape
        nu = (self.L @ torch.randn(n, k, generator=self.gen, dtype=DTYPE)) * self.prior.v.sqrt()
        R = H - self.prior.m
        u = float(torch.rand((), generator=self.gen, dtype=DTYPE))
        log_t = ll + math.log(max(u, 1e-300))
        theta = float(torch.rand((), generator=self.gen, dtype=DTYPE)) * TWO_PI
        lo, hi = theta - TWO_PI, theta
        for _ in range(200):
            Hn = self.prior.m + R * math.cos(theta) + nu * math.sin(theta)
            lln = self._loglik(Hn, y, sigma)
            if lln > log_t:
                return Hn, lln, True
            if theta < 0:
                lo = theta
            else:
                hi = theta
            theta = lo + (hi - lo) * float(torch.rand((), generator=self.gen, dtype=DTYPE))
        return H, ll, False

    def _level_set_moves(self, H, Q):
        """Random-scan MH over points; each proposal preserves g(h(x_i)) exactly,
        so the acceptance ratio is the GP prior ratio alone."""
        if not self.outer.moves:
            return H, 0, 0
        n, k = H.shape
        gH = self.outer.fn(H)
        acc = tried = 0
        for i in torch.randperm(n, generator=self.gen)[: self.moves_per_sweep].tolist():
            for move in self.outer.moves:
                prop = move(H[i].clone(), self.gen)
                if prop is None or not bool(torch.isfinite(prop).all()):
                    continue
                g_new = self.outer.fn(prop.unsqueeze(0))[0]
                if abs(float(g_new - gH[i])) > 1e-7 * (1.0 + abs(float(gH[i]))):
                    continue  # safety net: never accept a move that changes g
                tried += 1
                R = H - self.prior.m
                dlog = 0.0
                for j in range(k):
                    alpha = R[i, j]
                    beta = prop[j] - self.prior.m[j]
                    qr = Q[i] @ R[:, j]
                    dq = Q[i, i] * (beta ** 2 - alpha ** 2) + 2.0 * (beta - alpha) * (qr - Q[i, i] * alpha)
                    dlog += float(-0.5 * dq / self.prior.v[j])
                u = float(torch.rand((), generator=self.gen, dtype=DTYPE))
                if math.log(max(u, 1e-300)) < dlog:
                    H = H.clone()
                    H[i] = prop
                    gH = gH.clone()
                    gH[i] = g_new
                    acc += 1
        return H, acc, tried

    def _project(self, H, y, steps: int = 80, lr: float = 0.05):
        """Move latent rows onto the constraint surface {g(h) = y}.

        Elliptical slice sampling only takes steps of order sigma / sd(h) once
        the likelihood is tight, so a point initialized off the surface -- as a
        newly appended BO point is, sitting at its GP conditional mean -- never
        reaches it, and the error compounds over iterations.  Local descent
        lands on the preimage nearest that mean, which keeps whichever branch
        the prior prefers; the level-set moves then explore the other branches.
        """
        Z = H.clone().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((self.problem.g(Z) - y) ** 2).sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
        Z = Z.detach()
        return Z if torch.isfinite(Z).all() else H

    def _mcmc(self, X, y):
        n, k = len(X), self.prior.k
        Q = torch.cholesky_inverse(self.L)
        sd = self.prior.v.sqrt()

        warm = self.states is not None and self.states[0].shape[0] < n and \
            torch.equal(self._X_prev, X[: self.states[0].shape[0]])
        if warm:
            n_old = self.states[0].shape[0]
            states = []
            for Z in self.states:
                mean, _ = self.prior.conditional(X[:n_old], Z.unsqueeze(0), X[n_old:])
                new_rows = self._project(mean[0], y[n_old:])
                states.append(torch.cat([Z, new_rows], 0))
            # No annealing on a warm start.  The chain is already on the
            # posterior and only the appended point needs to settle; re-heating
            # sigma scatters latents that were already correct faster than the
            # short warm burn-in can recover them, and the error compounds over
            # BO iterations.
            burn, sigma0 = self.burn, self.noise
        else:
            states = [self._project(
                self.prior.m + (self.L @ torch.randn(n, k, generator=self.gen, dtype=DTYPE)) * sd, y)
                for _ in range(self.n_chains)]
            burn, sigma0 = self.burn_first, max(0.3, self.noise)

        anneal_len = max(1, burn // 2)
        chains, ess_ok, mv_acc, mv_try = [], 0, 0, 0
        for c, H in enumerate(states):
            kept = []
            for t in range(burn + self.keep):
                frac = min(1.0, t / anneal_len)
                sigma = sigma0 * (self.noise / sigma0) ** frac
                ll = self._loglik(H, y, sigma)
                H, ll, ok = self._ess_step(H, y, sigma, ll)
                H, a, tr = self._level_set_moves(H, Q)
                if t >= burn:
                    ess_ok += int(ok)
                    mv_acc, mv_try = mv_acc + a, mv_try + tr
                    if (t - burn) % self.thin == 0:
                        kept.append(H.clone())
            states[c] = H
            chains.append(torch.stack(kept))

        self.states, self._X_prev = states, X.clone()
        self.chain_F = chains
        self.F = torch.cat(chains, 0)
        n_post = max(1, self.keep * self.n_chains)
        self.stats = {
            "u_lost": self.F.var(0).mean().item(),
            "method": "mcmc",
            "ess_accept": ess_ok / n_post,
            "move_accept": (mv_acc / mv_try) if mv_try else float("nan"),
        }

    # ------------------------------------------------------------- predict
    def prepare_samples(self, n: int, seed: int):
        """Draw exactly n predictive samples, each pairing its own noise with a
        latent sample cycled from the chain.

        Giving every latent sample a *shared* noise draw instead collapses the
        Monte Carlo estimate whenever the latent posterior is tight: the
        samples then differ only by their posterior means, which are nearly
        identical, so the effective sample size falls to one.  Matching the
        composite arm's sample count here keeps the two comparable."""
        gen = torch.Generator().manual_seed(seed)
        self.eps = torch.randn(n, self.prior.k, generator=gen, dtype=DTYPE)
        self.idx = torch.arange(n) % self.F.shape[0]

    def sample_f(self, Xc):
        mean, sd = self.prior.conditional(self.X, self.F, Xc, self.L)   # (S,m,k), (m,k)
        Hs = mean[self.idx] + sd * self.eps[:, None, :]                  # (n,m,k)
        return self.problem.g(Hs)

    def diagnostics(self):
        return dict(self.stats)


ARMS = {"direct": DirectArm, "composite": CompositeArm, "latent": LatentArm}
