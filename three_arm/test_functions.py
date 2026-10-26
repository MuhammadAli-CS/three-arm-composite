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
def _spectral_sample(d, n_features, lengthscale, kernel, gen):
    """Draw frequencies from the kernel's spectral density.

    RBF's is Gaussian.  Matern-nu's is a multivariate Student-t with 2*nu
    degrees of freedom, sampled as a Gaussian scaled by an inverse Gamma --
    which is what makes Matern draws rougher than RBF ones.  BoTorch's default
    kernel is Matern 5/2, so generating h from it is what makes the surrogate
    correctly specified about the inner function rather than merely close.
    """
    z = torch.randn(n_features, d, generator=gen, dtype=DTYPE)
    if kernel == "rbf":
        return z / lengthscale
    # u ~ Gamma(nu, nu), so omega = z / (lengthscale * sqrt(u)) is Student-t
    # with 2*nu degrees of freedom.  torch.distributions.Gamma.sample() takes
    # no generator and reads the global RNG, so using it made every Matern
    # problem a DIFFERENT random function on each construction -- the draw
    # seed stopped selecting anything, maps no longer shared an h, and a
    # stored f_min could not be recomputed afterwards.
    #
    # Sampling it from the seeded generator instead, via the decomposition
    #     Gamma(2.5, 1) = Exp(1) + Exp(1) + Gamma(0.5, 1),
    # with Exp(1) = -log U and Gamma(0.5, 1) = Z^2 / 2.  Exact, and only needs
    # rand/randn, both of which accept the generator.
    nu = 2.5
    u1 = torch.rand(n_features, 1, generator=gen, dtype=DTYPE).clamp_min(1e-300)
    u2 = torch.rand(n_features, 1, generator=gen, dtype=DTYPE).clamp_min(1e-300)
    zh = torch.randn(n_features, 1, generator=gen, dtype=DTYPE)
    u = (-u1.log() - u2.log() + 0.5 * zh.square()) / nu
    return z / (lengthscale * u.sqrt())


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
                 n_features: int = 1024, n_ref: int = 16384,
                 kernel: str = "rbf"):
        gen = torch.Generator().manual_seed(seed)
        self.kernel = kernel
        self.d, self.lengthscale, self.seed = d, lengthscale, seed
        self.omega = _spectral_sample(d, n_features, lengthscale, kernel, gen)
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


def _poly_swap(roots):
    """Jump to another real preimage of the same value.

    Only needed if the latent arm is ever run on these maps; the composite vs
    direct comparison does not use it.  Roots of the shifted polynomial are
    found numerically, which is cheap at these degrees.
    """
    import numpy as np

    def move(h, gen):
        z = float(h[0])
        coef = np.poly(np.array(roots, dtype=float))          # monic, roots a_i
        target = float(np.polyval(coef, z))
        shifted = coef.copy(); shifted[-1] -= target
        rts = np.roots(shifted)
        real = [float(r.real) for r in rts
                if abs(r.imag) < 1e-8 and abs(r.real - z) > 1e-6]
        if not real:
            return None
        pick = real[int(torch.randint(len(real), (1,), generator=gen))]
        prop = h.clone(); prop[0] = pick
        return prop
    return move


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

    # ----------------------------------------------------------------
    # Bijectivity family: g(z) = prod_i (z - a_i).
    #
    # The proposition says a bijective g gives the composite model no
    # information advantage, whatever its nonlinearity.  Degree alone cannot
    # test that, because a higher-degree polynomial is also a wilder function
    # and harder for any GP to fit.  So each non-injective member is paired
    # with a monomial z^m of the SAME degree, which is strictly monotone for
    # odd m: same growth, same curvature scale, opposite injectivity.
    #
    # Roots are spread across the bulk of the standardized h (sd ~ 1.3) so
    # that the ambiguous region is actually visited.
    ROOTS = {1: [0.0], 2: [-1.0, 1.0], 3: [-1.5, 0.0, 1.5],
             4: [-1.5, -0.5, 0.5, 1.5], 5: [-2.0, -1.0, 0.0, 1.0, 2.0]}

    def _poly(a):
        def fn(y, a=a):
            z = y[..., 0]
            out = torch.ones_like(z)
            for r in a:
                out = out * (z - r)
            return out
        return fn

    for m, a in ROOTS.items():
        kind = "linear" if m == 1 else "lossy"
        add(OuterMap(f"root{m}", 1, _poly(a), kind,
                     "which of up to %d preimages" % m,
                     inverse=(lambda v: v) if m == 1 else None,
                     linear_coef=[1.0] if m == 1 else None,
                     moves=[] if m == 1 else [_poly_swap(a)]))

    # ----------------------------------------------------------------
    # Root-spread sweep: the clean knob on injectivity.
    #
    # mono_m and root_m are the same polynomial with the roots in different
    # places -- stacked at the origin, or spread out.  Sliding them together
    # therefore varies injectivity CONTINUOUSLY while holding the degree, the
    # leading coefficient and the tail behaviour fixed, which is exactly the
    # confound the matched pairs could not break on their own.
    #
    #     g_s(z) = prod_i (z - s a_i),    a = linspace(-2, 2, m)
    #
    # s = 0 collapses every root to the origin, giving z^m: injective for odd
    # m, and still two-to-one for even m, which is a property of even degree
    # rather than of the roots.
    for m in (2, 3, 4, 5, 6):
        base = [-2.0 + 4.0 * i / (m - 1) for i in range(m)]
        for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = [s_ * r for r in base]
            injective = (s_ == 0.0 and m % 2 == 1)
            add(OuterMap(f"spr{m}s{int(s_ * 100):03d}", 1, _poly(a),
                         "invertible" if injective else "lossy",
                         f"preimages at root spread {s_:g}",
                         inverse=(lambda v, m=m: v.sign() * v.abs() ** (1.0 / m))
                                 if injective else None,
                         moves=[] if injective else [_poly_swap(a)]))

    # Matched injective partners: z^m is strictly increasing for odd m, so it
    # is nonlinear, same degree, and loses nothing.
    for m in (3, 5):
        add(OuterMap(f"mono{m}", 1, lambda y, m=m: y[..., 0] ** m,
                     "invertible", "nothing (monotone, odd power)",
                     inverse=lambda v, m=m: v.sign() * v.abs() ** (1.0 / m)))

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
    # ----------------------------------------------------------------
    # Intermediate-count sweep: information loss WITHOUT folding.
    #
    # In one dimension non-injective means non-monotone, which means g folds,
    # which means y(x) gains turning points -- so destroying information and
    # making the objective harder to fit are the same operation and cannot be
    # told apart.  With k intermediates they separate:
    #
    #   sum_k    linear, so f is exactly a GP and Direct is correctly
    #            specified -- the misspecification channel is pinned at zero
    #            for every k, while observing h still reveals the split that
    #            y can never show.  Pure information.
    #   max_k    one kink, no folds: mild misspecification, same information.
    #   sq_k     sum of squares, which does fold: both channels active.
    #
    # k therefore moves information and the row moves misspecification,
    # independently.  k = 8 matches Truss, k = 6 Color, k = 29 power flow.
    for k in (2, 4, 8, 16):
        add(OuterMap(f"sum_k{k}", k, lambda y: y.sum(-1), "linear",
                     f"the split of a sum over {k} intermediates",
                     linear_coef=[1.0] * k, moves=[_sum_shift] if k == 2 else []))
        add(OuterMap(f"max_k{k}", k, lambda y: y.max(-1).values, "lossy",
                     f"every value but the largest of {k}"))
        add(OuterMap(f"sq_k{k}", k, lambda y: (y ** 2).sum(-1), "lossy",
                     f"the sign of each of {k} intermediates"))

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
    # Composite vs Direct only: does bijectivity, rather than nonlinearity,
    # decide the gain?  root1 is linear; mono3/mono5 are nonlinear but
    # injective; root2..root5 are non-injective at matched degrees.
    "bijectivity": ["root1", "mono3", "mono5", "exp_b1", "sinh_b1",
                    "root2", "root3", "root4", "root5"],
    # Degree held fixed, injectivity swept by sliding the roots together.
    "spread": [f"spr{m}s{s:03d}" for m in (2, 3, 4, 5, 6)
               for s in (0, 25, 50, 75, 100)],
    # Information loss without folding: k moves information, the family moves
    # misspecification.  The linear row is the decisive one.
    "kcount": [f"{fam}_k{k}" for fam in ("sum", "max", "sq")
               for k in (2, 4, 8, 16)],
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
                 draw_seed: int = 0, n_ref: int = 65536, n_features: int = 1024,
                 kernel: str = "rbf"):
        self.outer = OUTER_MAPS[map_name]
        self.name, self.d, self.lengthscale = map_name, d, lengthscale
        self.kernel = kernel
        self.draws = [RFFDraw(d, lengthscale, seed=1000 * draw_seed + j,
                              n_features=n_features, kernel=kernel)
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
        """Reference optimum, used only as the subtrahend in the regret.

        Descending f = g(h(x)) by gradient fails wherever g is flat or kinked:
        clip and relu have zero gradient over a whole region, so Adam never
        leaves its starting point and the estimate stays at the best reference
        sample.  That understates the optimum, and a BO run that then finds
        something better produces a NEGATIVE regret -- which is how clip_c2
        came out at -1.4e-2.

        For a single intermediate there is no need to differentiate g at all.
        The domain is connected and h continuous, so h([0,1]^d) is an interval,
        and
            min_x g(h(x)) = min over that interval of g,
        which a dense grid plus local refinement finds exactly, for any g.
        Only h has to be optimised by gradient, and h is a smooth RFF sum.
        """
        with torch.no_grad():
            vals = torch.cat([self.f(ref[i:i + _CHUNK]) for i in range(0, len(ref), _CHUNK)])
        best = vals.min().item()

        # gradient polish on f: still the only option when k > 1
        X = ref[vals.argsort()[:n_starts]].clone().requires_grad_(True)
        opt = torch.optim.Adam([X], lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            self.f(X).sum().backward()
            opt.step()
            with torch.no_grad():
                X.clamp_(0.0, 1.0)
        with torch.no_grad():
            best = min(best, self.f(X).min().item())

        if self.outer.k == 1:
            best = min(best, self._min_over_h_range(ref))
        return best

    def _min_over_h_range(self, ref: torch.Tensor, grid: int = 400001,
                          n_starts: int = 20, steps: int = 400) -> float:
        """min of g over the interval h actually reaches, found on a grid."""
        with torch.no_grad():
            hv = torch.cat([self.h(ref[i:i + _CHUNK])[:, 0]
                            for i in range(0, len(ref), _CHUNK)])
        lo, hi = float(hv.min()), float(hv.max())
        # push the endpoints out by gradient on h itself, which is smooth
        for sign in (1.0, -1.0):
            idx = (hv * sign).argsort()[:n_starts]
            X = ref[idx].clone().requires_grad_(True)
            opt = torch.optim.Adam([X], lr=0.01)
            for _ in range(steps):
                opt.zero_grad()
                (sign * self.h(X)[:, 0]).sum().backward()
                opt.step()
                with torch.no_grad():
                    X.clamp_(0.0, 1.0)
            with torch.no_grad():
                v = self.h(X)[:, 0]
                lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))

        with torch.no_grad():
            z = torch.linspace(lo, hi, grid, dtype=DTYPE)
            gz = self.g(z.unsqueeze(-1))
            j = int(gz.argmin())
            # refine on the two neighbouring cells
            a = z[max(j - 1, 0)].item()
            b = z[min(j + 1, grid - 1)].item()
            zf = torch.linspace(a, b, 20001, dtype=DTYPE)
            return min(float(gz[j]), float(self.g(zf.unsqueeze(-1)).min()))

    def describe(self) -> str:
        o = self.outer
        tail = f", analog of {o.analog}" if o.analog else ""
        return f"{self.name}: k={o.k}, {o.kind}, discards {o.discards}{tail}"


def make_problem(map_name: str, **kwargs) -> Problem:
    return Problem(map_name, **kwargs)
