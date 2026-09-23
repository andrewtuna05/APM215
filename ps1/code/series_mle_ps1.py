"""P-Set 1, Part 1 -- estimating p from World Series data (item 1.3).

This file is already named what you submit. Fill in the FOUR function bodies marked
`raise NotImplementedError`: the outcome probabilities (your 1.1.a formula), the negative
log-likelihood, the numerical MLE, and the series simulator. `load_counts` and `main`
are supplied: read them, do not rewrite them. Keep every name, argument order and return
type exactly as given; a TF runs this file.

Run it from the unpacked zip's top directory (the one holding code/ and data/):

    python3 code/series_mle_ps1.py data/ws_1905_1951.csv

It prints p-hat and the replicate standard deviations, and writes nll_series.png next to
this file. Needs Python 3.10+ with numpy, scipy and matplotlib -- as for P-Set 0.
"""

import sys
import pathlib

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import comb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
# The unpacked Canvas zip has data/ next to code/; the course repo has data/public/.
_CANDIDATES = [HERE.parent / "data" / "ws_1905_1951.csv",
               HERE.parent / "data" / "public" / "ws_1905_1951.csv"]
DEFAULT_DATA = next((c for c in _CANDIDATES if c.exists()), _CANDIDATES[0])

def prob_loser_wins(p):
    """P(K = k | p) for k = 0..3 in a best-of-7: C(3+k,k) [p^4 q^k + q^4 p^k].

    p is the probability the BETTER team wins one game (p >= 1/2); the loser of the series
    may be either team, which is why both terms appear.
    """
    raise NotImplementedError("TODO")

def nll_series(p, counts):
    """Negative log-likelihood of the counts n_0..n_3 under P(K = k | p).

    `counts` is a length-4 array: how many series the losing team won 0, 1, 2, 3 games.
    The multinomial coefficient is dropped (it does not depend on p). Items 1.1.c and 1.3.a.
    """
    raise NotImplementedError("TODO")

def mle_series(counts):
    """Item 1.3.b: MLE of p by numerical minimisation of nll_series on [1/2, 1).

    Bounded scalar minimisation, deliberately. Item 1.1.d shows l(p) = l(1 - p), so
    dl/dp = 0 exactly at p = 1/2 -- and a gradient method such as L-BFGS-B with a lower
    bound of 0.5 can land on that bound after its first step, see a zero projected
    gradient, and report p = 0.5 as converged. (It did, from x0 = 0.75, on this data.)
    Brent's bounded method never evaluates the gradient, so it cannot be fooled that way.
    """
    raise NotImplementedError("TODO")

def simulate_series(p, n_series, rng):
    """Simulate `n_series` best-of-7 series at better-team win probability p.

    Returns the counts n_0..n_3 of how many series the losing team won 0..3 games.
    Plays each game as a Bernoulli(p) trial for the better team until one side has 4.
    """
    raise NotImplementedError("TODO")

def load_counts(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.bincount(np.asarray(data["loser_wins"], dtype=np.intp), minlength=4)[:4]


def main(path=DEFAULT_DATA):
    counts = load_counts(path)
    n = int(counts.sum())
    print(f"data: n = {n} series, counts n_0..n_3 = {counts.tolist()}")

    p_hat = mle_series(counts)
    print(f"p-hat = {p_hat:.4f}   NLL = {nll_series(p_hat, counts):.3f}   "
          f"(Mosteller: 0.65, NLL {nll_series(0.65, counts):.3f})")
    pred = n * prob_loser_wins(p_hat)
    print("predicted counts at p-hat:", np.round(pred, 2).tolist(), " observed:", counts.tolist())

    # 1.3.b -- the log-likelihood curve
    grid = np.linspace(0.5, 0.999, 400)
    ll = [-nll_series(g, counts) for g in grid]
    plt.figure(figsize=(6, 3.6))
    plt.plot(grid, ll)
    plt.axvline(p_hat, ls="--", label=f"$\\hat p$ = {p_hat:.3f}")
    plt.xlabel("p (better team's single-game win probability)")
    plt.ylabel("log-likelihood")
    plt.title("World Series 1905-1951: log-likelihood of p")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "nll_series.png", dpi=120)

    # 1.3.c -- how well do 44 (and 4400) series pin p down? (interpreted in 1.4.b)
    rng = np.random.default_rng(0)
    for n_series in (44, 4400):
        est = np.array([mle_series(simulate_series(p_hat, n_series, rng)) for _ in range(2000)])
        print(f"N = {n_series:>5}: SD of p-hat over 2000 replicates = {est.std(ddof=1):.4f}  "
              f"(mean {est.mean():.4f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
