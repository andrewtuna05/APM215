"""
Directional forecast evaluation for a stock index.

This script:
1. estimates the historical up-market probability p,
2. computes a no-information benchmark using the forecaster's own call mix,
3. performs an exact one-sided binomial test,
4. reports a Wilson confidence interval,
5. estimates the probability of observing a streak at least as long as the
   forecaster's best streak somewhere in n predictions.

Input format
------------
prices.csv:
    date,close

forecasts.csv:
    date,call

where call is one of:
    up, down

The forecast dated t is interpreted as a prediction of the close h trading
days later relative to the close on t.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "close"]].dropna().sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("prices.csv contains duplicate dates.")
    return df.reset_index(drop=True)


def load_forecasts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "call"]].dropna().sort_values("date")
    df["call"] = df["call"].str.lower().str.strip()
    bad = ~df["call"].isin(["up", "down"])
    if bad.any():
        raise ValueError("Forecast calls must be 'up' or 'down'.")
    if df["date"].duplicated().any():
        raise ValueError("forecasts.csv contains duplicate forecast dates.")
    return df.reset_index(drop=True)


def add_horizon_outcomes(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1 trading day.")

    x = prices.copy()
    x["future_close"] = x["close"].shift(-horizon)
    x["outcome"] = np.where(
        x["future_close"] > x["close"], "up",
        np.where(x["future_close"] < x["close"], "down", "flat")
    )
    x.loc[x["future_close"].isna(), "outcome"] = np.nan
    return x


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / denom
    return center - half, center + half


def longest_streak(correct: np.ndarray) -> int:
    best = current = 0
    for x in correct:
        if x:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def simulate_streak_probability(
    n: int,
    q0: float,
    target_streak: int,
    simulations: int = 100_000,
    seed: int = 0,
) -> float:
    if target_streak <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    hits = 0

    # Chunking avoids allocating an unnecessarily huge matrix.
    chunk = 10_000
    completed = 0

    while completed < simulations:
        m = min(chunk, simulations - completed)
        draws = rng.random((m, n)) < q0

        for row in draws:
            if longest_streak(row) >= target_streak:
                hits += 1

        completed += m

    return hits / simulations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--forecasts", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--baseline-end",
        required=True,
        help="Last date used to estimate market up probability, YYYY-MM-DD.",
    )
    parser.add_argument("--simulations", type=int, default=100_000)
    args = parser.parse_args()

    prices = add_horizon_outcomes(load_prices(args.prices), args.horizon)
    forecasts = load_forecasts(args.forecasts)

    baseline_end = pd.Timestamp(args.baseline_end)

    # Estimate p using only data available before evaluation.
    baseline = prices[
        (prices["date"] <= baseline_end) &
        (prices["outcome"].isin(["up", "down"]))
    ].copy()

    if len(baseline) == 0:
        raise ValueError("No usable observations in the baseline period.")

    p_up = (baseline["outcome"] == "up").mean()

    # Merge calls with realized outcomes.
    evaluation = forecasts.merge(
        prices[["date", "outcome"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    # Remove unresolved and flat outcomes.
    evaluation = evaluation[evaluation["outcome"].isin(["up", "down"])].copy()

    if len(evaluation) == 0:
        raise ValueError("No forecasts could be matched to resolved outcomes.")

    evaluation["correct"] = evaluation["call"] == evaluation["outcome"]

    n = len(evaluation)
    k = int(evaluation["correct"].sum())
    accuracy = k / n
    r_up_calls = (evaluation["call"] == "up").mean()

    # Null accuracy preserving the forecaster's tendency to say up/down.
    q0 = r_up_calls * p_up + (1 - r_up_calls) * (1 - p_up)

    test = binomtest(k, n=n, p=q0, alternative="greater")
    ci_low, ci_high = wilson_interval(k, n)

    streak = longest_streak(evaluation["correct"].to_numpy())
    streak_prob = simulate_streak_probability(
        n=n,
        q0=q0,
        target_streak=streak,
        simulations=args.simulations,
    )

    print(f"Forecast horizon: {args.horizon} trading day(s)")
    print(f"Baseline observations: {len(baseline)}")
    print(f"Estimated market up probability p: {p_up:.4f}")
    print(f"Evaluation forecasts n: {n}")
    print(f"Up-call fraction r: {r_up_calls:.4f}")
    print(f"No-information expected accuracy q0: {q0:.4f}")
    print(f"Observed correct calls: {k}/{n}")
    print(f"Observed accuracy: {accuracy:.4f}")
    print(f"95% Wilson interval: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"One-sided exact binomial p-value: {test.pvalue:.6g}")
    print(f"Longest observed correct streak: {streak}")
    print(
        f"Estimated P(at least one streak >= {streak} in {n} calls | null): "
        f"{streak_prob:.6g}"
    )


if __name__ == "__main__":
    main()
