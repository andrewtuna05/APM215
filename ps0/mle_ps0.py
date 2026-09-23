"""P-Set 0, Part 4.

This file is already named what you submit. Fill in the THREE function bodies marked
`raise NotImplementedError` — the likelihood, the closed-form estimate, and the
numerical one — and submit this same file. The other two — `standard_error_experiment` and `estimate_slope` —
are supplied already working. Read them; do not rewrite them.

Signatures, argument order and return types must stay exactly as given: your
submission is checked automatically.

The checks are diagnostic. Credit is for a genuine attempt, not for passing them all —
a wrong answer passes the completion check; a stub that returns nothing does not.

Setup. You need Python 3.10+ with numpy, scipy and matplotlib:

    # macOS / Linux
    python3 -m venv ps0_env && source ps0_env/bin/activate

    # Windows (PowerShell)
    py -m venv ps0_env; .\\ps0_env\\Scripts\\Activate.ps1

    pip install numpy scipy matplotlib

Then run it -- macOS/Linux `python3 mle_ps0.py`, Windows `python mle_ps0.py`. A standard
Windows install provides `py` and `python` but usually not `python3`, so the same command
does not work on both.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")  # write the figure to a file without needing a display
import matplotlib.pyplot as plt  # noqa: E402


def nll_exponential(lam, data):
    """Negative log-likelihood of `data` under Exponential(lam).

    Parameters
    ----------
    lam : float
        Rate parameter, lam > 0. NOTE: scipy.optimize.minimize passes this as a
        length-1 array rather than a scalar, so unpack it defensively.
    data : np.ndarray
        1-D array of non-negative floats.

    Returns
    -------
    float
        The value of -l(lam). Item 3.1 gives you the formula for l(lam).
    """
    lam = float(np.asarray(lam).reshape(-1)[0])
    x = np.asarray(data, dtype=float)
    # raise NotImplementedError("TODO")
    return float(-len(x) * np.log(lam) + lam * np.sum(x))


def mle_exponential_analytic(data):
    """Closed-form MLE of the exponential rate.

    Returns
    -------
    float
        Your closed-form result from item 3.1.
    """
    x = np.asarray(data, dtype=float)
    # raise NotImplementedError("TODO")
    S = float(np.mean(x))
    return (1 / S)


def mle_exponential_numeric(data):
    """MLE of the exponential rate, by minimizing nll_exponential numerically.

    Use `scipy.optimize.minimize` with `bounds` keeping lam strictly positive —
    the log-likelihood is undefined at lam = 0.

    Use a FIXED starting guess such as 1.0. Do not start at 1/data.mean(): that is
    the analytic answer, so the optimizer would hand back its own starting point and
    the agreement you discuss in item 4.3 would be true by construction rather than
    earned. The two routes have to be independent for the comparison to mean anything.

    Returns
    -------
    float
    """
    x = np.asarray(data, dtype=float)
    # raise NotImplementedError("TODO")

    def n_log_like_dist(x0):
        return -1.0*(np.sum(np.log(x0) - x0*x))

    lam = minimize(n_log_like_dist, x0 = [1.0], bounds = [(1e-12, None)])
    return lam.x[0]


def standard_error_experiment(sample_sizes, lam_true, n_trials, rng):
    """How the spread of the estimator shrinks as the sample grows.

    SUPPLIED — you do not need to write this one. It is the Monte Carlo loop: for each
    n, draw `n_trials` datasets of size n, estimate lam for each, and record the spread
    of those estimates. It calls YOUR `mle_exponential_analytic`, so it only works once
    that one does.

    Read it rather than skipping it — the pattern (simulate many datasets, look at the
    distribution of an estimator) is the whole of Monte Carlo error analysis, and week 2
    formalises it.

    Note `rng.exponential` takes the SCALE (1/lam), not the rate.

    Parameters
    ----------
    sample_sizes : list[int]
    lam_true : float
    n_trials : int
    rng : np.random.Generator

    Returns
    -------
    dict[int, float]
        Maps each n to the standard deviation of the estimates at that n.
    """
    out = {}
    for n in sample_sizes:
        samples = rng.exponential(1.0 / lam_true, size=(n_trials, int(n)))
        estimates = np.array([mle_exponential_analytic(row) for row in samples])
        sd = float(np.std(estimates))
        if sd == 0.0:
            # Every one of n_trials different datasets gave the SAME estimate, which a
            # real estimator cannot do. Say so here rather than letting it surface later
            # as log(0) -> a nan slope and an empty log-log plot, which looks like our
            # bug and is actually a signal about mle_exponential_analytic.
            raise ValueError(
                f"At n={int(n)}, all {n_trials} simulated datasets produced the identical "
                f"estimate {estimates[0]!r}. That means mle_exponential_analytic is "
                "ignoring its `data` argument -- check it uses the sample, not a constant."
            )
        out[int(n)] = sd
    return out


def estimate_slope(sample_sizes, std_devs):
    """Fit log(sd) against log(n) and return the slope.

    SUPPLIED — you do not need to write this one. It is here because the log-log
    fit is the standard way to measure a power law numerically, and week 2
    asks you to interpret the number it returns.

    Note the value lookup: `standard_error_experiment` returns a dict, and
    iterating a dict yields its KEYS (the sample sizes), not its values -- so
    `list(std_devs)` would fit log(n) against log(n) and return +1.

    Parameters
    ----------
    sample_sizes : list[int]
    std_devs : dict[int, float] | list[float]
        Either the dict from standard_error_experiment, or the standard
        deviations in the same order as sample_sizes.

    Returns
    -------
    float
        The fitted slope. Theory predicts a specific value — week 2 asks you
        whether your measurement is consistent with it.
    """
    sizes = np.asarray(sample_sizes, dtype=float)
    if isinstance(std_devs, dict):
        sds = np.asarray([std_devs[n] for n in sample_sizes], dtype=float)
    else:
        sds = np.asarray(std_devs, dtype=float)
    slope, _intercept = np.polyfit(np.log(sizes), np.log(sds), 1)
    return float(slope)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lam_true = 2.0

    # One dataset: compare the two routes.
    data = rng.exponential(1.0 / lam_true, size=500)
    analytic = mle_exponential_analytic(data)
    numeric = mle_exponential_numeric(data)
    print(f"analytic : {analytic:.6f}")
    print(f"numeric  : {numeric:.6f}")
    print(f"|diff|   : {abs(analytic - numeric):.3e}")

    # The scaling sweep.
    sample_sizes = [100, 300, 1000, 3000, 10000]
    sds = standard_error_experiment(sample_sizes, lam_true, n_trials=500, rng=rng)
    slope = estimate_slope(sample_sizes, sds)
    print(f"fitted slope: {slope:.4f}")

    # The log-log plot. Axis labels and a title are required.
    plt.figure()
    plt.loglog(sample_sizes, [sds[n] for n in sample_sizes], "o-")
    plt.xlabel("sample size n")
    plt.ylabel("standard deviation of lambda-hat")
    plt.title(f"Standard error scaling (slope = {slope:.3f})")
    plt.savefig("se_scaling.png", dpi=150, bbox_inches="tight")
    print("wrote se_scaling.png")
