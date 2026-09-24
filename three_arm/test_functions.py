"""Synthetic test functions for the three-arm composite experiment.

Refines Experiment 1 of the future-directions doc.  Every intermediate h_j is a
fixed RBF-GP draw from the common random-Fourier-feature generator (doc
Section 4), and only the outer map g changes.  The maps are chosen by *what
they discard*, so the three arms separate:

    Arm 1  Direct     GP on y                          knows neither g nor h
    Arm 2  Composite  GP on the observed h, apply g    knows g, observes h
    Arm 3  Latent     GP prior on h, observe only y    knows g only

Arm 3 - Arm 1 is the value of knowing g; Arm 2 - Arm 3 is the value of
observing h, which is zero by construction whenever g is invertible.

Each OuterMap also carries *level-set moves*: proposals h -> h' with
g(h') == g(h) exactly and a symmetric proposal density.  Arm 3 uses them to
move along the ambiguity that g creates (see arms.LatentArm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch.quasirandom import SobolEngine

DTYPE = torch.double
TWO_PI = 2.0 * math.pi
_CHUNK = 4096


# --------------------------------------------------------------------------
# Common GP-draw generator (doc Section 4)
# --------------------------------------------------------------------------
class RFFDraw:
    """Fixed RBF-GP draw via random Fourier features.

    h(x) = sqrt(2/M) sum_m a_m cos(w_m^T x + b_m) with a ~ N(0,1),
    b ~ U(0, 2pi), w ~ N(0, l^{-2} I), whose prior is GP(0, k_RBF) with unit
    outputscale.  The draw is standardized on a Sobol reference set as the doc
    specifies, so the implied prior on the standardized function is
    GP(-mu/sd, k_RBF / sd^2).  That prior is exposed as ``prior_mean`` and
    ``prior_var`` and is the known prior shared by the Composite and Latent arms.
    """

    def __init__(self, d: int, lengthscale: float = 0.5, seed: int = 0,
                 n_features: int = 1024, n_ref: int = 16384):
        gen = torch.Generator().manual_seed(seed)
        self.d, self.lengthscale, self.seed = d, lengthscale, seed
        self.omega = torch.randn(n_features, d, generator=gen, dtype=DTYPE) / lengthscale
        self.b = TWO_PI * torch.rand(n_features, generator=gen, dtype=DTYPE)
        self.a = torch.randn(n_features, generator=gen, dtype=DTYPE)
        self.scale = math.sqrt(2.0 / n_features)
        ref = SobolEngine(d, scramble=True, seed=7919 + seed).draw(n_ref).to(DTYPE)
        raw = self._raw(ref)
        self.mu, self.sd = raw.mean().item(), raw.std().item()

    def _raw(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[0] <= _CHUNK:
            return self.scale * (torch.cos(X @ self.omega.T + self.b) @ self.a)
        return torch.cat([self._raw(X[i:i + _CHUNK]) for i in range(0, X.shape[0], _CHUNK)])

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (self._raw(X) - self.mu) / self.sd

    @property
    def prior_mean(self) -> float:
        return -self.mu / self.sd

    @property
    def prior_var(self) -> float:
        return 1.0 / self.sd ** 2


# --------------------------------------------------------------------------
# Level-set moves: g(move(h)) == g(h), symmetric proposal density
# --------------------------------------------------------------------------
def _pick(values, gen):
    return values[int(torch.randint(len(values), (1,), generator=gen))]


def _randn(gen):
    return torch.randn((), generator=gen, dtype=DTYPE)


def _reflect(center: float):
    """Involution h -> 2c - h: preserves any g symmetric about c."""
    def move(h, gen):
        return 2.0 * center - h
    return move


def _cos_moves(w: float):
    period = TWO_PI / w

    def reflect(h, gen):
        # through the nearest multiple of pi/w -- an involution that preserves cos(w h)
        a = torch.round(h * w / math.pi) * math.pi / w
        return 2.0 * a - h

    def translate(h, gen):
        return h + (period if torch.rand((), generator=gen) < 0.5 else -period)

    return [reflect, translate]


def _relu_walk(h, gen):
    if h[0] > 0:  # y > 0 pins h exactly
        return None
    prop = h + _pick([0.05, 0.3, 1.0], gen) * _randn(gen)
    return prop if prop[0] <= 0 else None


def _clip_pair_walk(c: float):
    """Move the clipped coordinate within its saturated region; the second
    coordinate, which carries the optimum, is left alone."""
    def move(h, gen):
        s = c * h[0]
        if abs(float(s)) < 1.0:  # unsaturated: pinned by the observation
            return None
        prop = h.clone()
        prop[0] = prop[0] + _pick([0.05, 0.3, 1.0], gen) * _randn(gen)
        same_side = (c * prop[0] >= 1.0) if s >= 1.0 else (c * prop[0] <= -1.0)
        return prop if same_side else None
    return move


def _sum_shift(h, gen):
    delta = _pick([0.05, 0.3, 1.0], gen) * _randn(gen)
    return torch.stack([h[0] + delta, h[1] - delta])


def _negate_pair(h, gen):
    return -h


def _product_scale(h, gen):
    t = torch.exp(_pick([0.05, 0.3, 1.0], gen) * _randn(gen))
    return torch.stack([h[0] * t, h[1] / t])


def _rotate(h, gen):
    phi = _pick([0.05, 0.5, math.pi], gen) * _randn(gen)
    c, s = torch.cos(phi), torch.sin(phi)
    return torch.stack([c * h[0] - s * h[1], s * h[0] + c * h[1]])


def _swap(h, gen):
    return h.flip(0)


def _max_walk(h, gen):
    small = 1 if h[0] > h[1] else 0
    prop = h.clone()
    prop[small] = prop[small] + _pick([0.05, 0.3, 1.0], gen) * _randn(gen)
    return prop if prop[small] < prop[1 - small] else None


# --------------------------------------------------------------------------
# Outer maps
# --------------------------------------------------------------------------
@dataclass
class OuterMap:
    name: str
    k: int                                    # number of intermediate inputs
    fn: Callable[[torch.Tensor], torch.Tensor]   # (..., k) -> (...)
    kind: str                                 # identity | invertible | linear | lossy
    discards: str
    analog: str = ""
    inverse: Optional[Callable] = None        # scalar invertible maps only
    linear_coef: Optional[List[float]] = None
    moves: List[Callable] = field(default_factory=list)


def _build_maps() -> Dict[str, OuterMap]:
    maps: Dict[str, OuterMap] = {}

    def add(m: OuterMap):
        maps[m.name] = m

    add(OuterMap("identity", 1, lambda y: y[..., 0], "identity", "nothing",
                 inverse=lambda v: v))

    for b in (0.5, 1.0, 2.0):
        add(OuterMap(f"exp_b{b:g}", 1, lambda y, b=b: torch.exp(b * y[..., 0]),
                     "invertible", "nothing (monotone, positive)",
                     inverse=lambda v, b=b: torch.log(v) / b))
        add(OuterMap(f"sinh_b{b:g}", 1, lambda y, b=b: torch.sinh(b * y[..., 0]),
                     "invertible", "nothing (monotone, unbounded)",
                     inverse=lambda v, b=b: torch.asinh(v) / b))

    # headline sweep: same form, information loss controlled by c
    for c in (0.0, 0.5, 1.0, 2.0, 3.0):
        add(OuterMap(f"shiftsq_c{c:g}", 1, lambda y, c=c: (y[..., 0] - c) ** 2,
                     "lossy", "sign of h - c", moves=[_reflect(c)]))

    add(OuterMap("abs", 1, lambda y: y[..., 0].abs(), "lossy",
                 "sign of h (with a kink)", moves=[_reflect(0.0)]))

    for w in (1.0, 2.0, 4.0):
        add(OuterMap(f"cos_w{w:g}", 1, lambda y, w=w: torch.cos(w * y[..., 0]),
                     "lossy", "which of many preimages", moves=_cos_moves(w)))

    # Minimizing -relu, not relu: relu itself is minimized on the whole censored
    # region h <= 0, which the initial design already contains, so every arm
    # scores a perfect regret and the condition carries no signal.  Negating
    # puts the optimum at max h, which is unique, while censoring the same set.
    # Note the optimum then sits where g is injective, so this condition tests
    # whether information destroyed *away from* the optimum matters at all.
    add(OuterMap("relu", 1, lambda y: -y[..., 0].clamp_min(0.0), "lossy",
                 "everything below zero (censoring, away from the optimum)",
                 moves=[_relu_walk]))

    # Likewise a bare clip is flat at both ends, so the optimum is a region.
    # Adding a second, unclipped intermediate makes the optimum a genuine
    # tradeoff while the first coordinate stays censored inside saturation.
    add(OuterMap("clip_c2", 2,
                 lambda y: (2.0 * y[..., 0]).clamp(-1.0, 1.0) + y[..., 1], "lossy",
                 "the clipped coordinate inside the saturated regions",
                 moves=[_clip_pair_walk(2.0)]))

    # two inputs -> one output: analogs of the real benchmarks
    add(OuterMap("sum", 2, lambda y: y[..., 0] + y[..., 1], "linear",
                 "how the sum splits", analog="Truss / RCM40",
                 linear_coef=[1.0, 1.0], moves=[_sum_shift]))
    add(OuterMap("product", 2, lambda y: y[..., 0] * y[..., 1], "lossy",
                 "how the product factors, and sign", analog="DTLZ2",
                 moves=[_negate_pair, _product_scale]))
    add(OuterMap("norm", 2, lambda y: torch.sqrt(y[..., 0] ** 2 + y[..., 1] ** 2 + 1e-12),
                 "lossy", "direction", analog="Color", moves=[_rotate]))
    add(OuterMap("max", 2, lambda y: torch.maximum(y[..., 0], y[..., 1]), "lossy",
                 "the smaller value", analog="Lake", moves=[_swap, _max_walk]))
    add(OuterMap("ratio", 2,
                 lambda y: torch.sigmoid(y[..., 0])
                 / (0.05 + torch.sigmoid(y[..., 0]) + torch.sigmoid(y[..., 1])),
                 "lossy", "scale", analog="RGB"))  # no exact moves: ESS only
    return maps


OUTER_MAPS: Dict[str, OuterMap] = _build_maps()

GROUPS: Dict[str, List[str]] = {
    "validate": ["identity", "sum", "exp_b1"],
    "sweep": ["shiftsq_c0", "shiftsq_c0.5", "shiftsq_c1", "shiftsq_c2", "shiftsq_c3"],
    "pairs": ["shiftsq_c0", "abs", "exp_b1", "sinh_b1",
              "cos_w1", "cos_w2", "cos_w4", "relu", "clip_c2"],
    "analogs": ["sum", "product", "norm", "max", "ratio"],
}


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------
class Problem:
    """Single-objective problem  f(x) = normalize(g(h(x))), minimized.

    The first intermediate uses the same draw seed in every condition, so the
    underlying functions are shared across all outer maps, as in Experiment 1.
    Normalization follows doc Section 8.5 (1st / 99th percentiles on a fixed
    Sobol reference set) and is part of the known outer map.
    """

    def __init__(self, map_name: str, d: int = 6, lengthscale: float = 0.5,
                 draw_seed: int = 0, n_ref: int = 65536, n_features: int = 1024):
        self.outer = OUTER_MAPS[map_name]
        self.name, self.d, self.lengthscale = map_name, d, lengthscale
        self.draws = [RFFDraw(d, lengthscale, seed=1000 * draw_seed + j, n_features=n_features)
                      for j in range(self.outer.k)]
        ref = SobolEngine(d, scramble=True, seed=31337).draw(n_ref).to(DTYPE)
        raw = self.outer.fn(self.h(ref))
        self.q01 = torch.quantile(raw, 0.01).item()
        self.q99 = torch.quantile(raw, 0.99).item()
        self.f_min = self._estimate_min(ref)

    # oracle ----------------------------------------------------------------
    def h(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([dr(X) for dr in self.draws], dim=-1)

    def g(self, H: torch.Tensor) -> torch.Tensor:
        return (self.outer.fn(H) - self.q01) / (self.q99 - self.q01)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        return self.g(self.h(X))

    def h_from_y(self, y: torch.Tensor) -> torch.Tensor:
        """Invert the full known map; only valid for identity / invertible kinds."""
        if self.outer.inverse is None:
            raise ValueError(f"{self.name} is not invertible")
        v = y * (self.q99 - self.q01) + self.q01
        return self.outer.inverse(v).unsqueeze(-1)

    # known prior -------------------------------------------------------------
    @property
    def prior_means(self) -> List[float]:
        return [dr.prior_mean for dr in self.draws]

    @property
    def prior_vars(self) -> List[float]:
        return [dr.prior_var for dr in self.draws]

    def _estimate_min(self, ref: torch.Tensor, n_starts: int = 20, steps: int = 200) -> float:
        with torch.no_grad():
            vals = torch.cat([self.f(ref[i:i + _CHUNK]) for i in range(0, len(ref), _CHUNK)])
        best = vals.min().item()
        X = ref[vals.argsort()[:n_starts]].clone().requires_grad_(True)
        opt = torch.optim.Adam([X], lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            self.f(X).sum().backward()
            opt.step()
            with torch.no_grad():
                X.clamp_(0.0, 1.0)
        with torch.no_grad():
            return min(best, self.f(X).min().item())

    def describe(self) -> str:
        o = self.outer
        tail = f", analog of {o.analog}" if o.analog else ""
        return f"{self.name}: k={o.k}, {o.kind}, discards {o.discards}{tail}"


def make_problem(map_name: str, **kwargs) -> Problem:
    return Problem(map_name, **kwargs)
