"""P-Set 1, Part 3 -- random walks (item 3.2).

This file is already named what you submit. Fill in the TWO function bodies marked
`raise NotImplementedError`: the walk simulator and the step variance it needs. The
table driver `moments_table` is supplied and `main` writes its output to moments.txt;
read them -- the standard-error formulas item 3.2 quotes are in `moments_table`.

Run it from anywhere:

    python3 code/random_walk_ps1.py

It writes moments.txt next to this file.
"""

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def simulate_walks(n_steps, n_walkers, delta, rng, step="pm"):
    """Positions X_0..X_n for `n_walkers` independent walks; shape (n_walkers, n_steps + 1).

    step="pm":      each step is +delta or -delta with probability 1/2
    step="uniform": each step is uniform on [-delta, delta]
    All steps are drawn in one call; there is no loop over walkers.
    """

    # create a position matrix thats n_walkers x n_steps+1 (to include 0)
    # each row represents one independent walk
    if step == "pm":
        steps = delta * rng.choice([-1, 1], size = (n_walkers, n_steps), p=[0.5, 0.5])

    elif step == "uniform":
        steps = rng.uniform(low=-1*delta, high=delta, size=(n_walkers, n_steps))

    positions = np.zeros([n_walkers, n_steps+1])

    for i in range(n_steps):
        positions[:, i+1] = positions[:, i] + steps[:, i]

    # raise NotImplementedError("TODO")
    return positions
    

def step_variance(delta, step):
    """Variance of ONE step: delta^2 for "pm", and your item 3.1.b answer for "uniform"."""
    if step == "pm":
        return np.pow(delta,2)
    elif step == "uniform" :
        return np.pow(delta,2)/3
    # raise NotImplementedError("TODO")
    

def moments_table(delta=1.0, M=10_000, n_max=1000, checkpoints=(10, 100, 1000), seed=0):
    rng = np.random.default_rng(seed)
    lines = [f"{'walk':8s} {'n':>5s} {'mean':>8s} {'se':>7s} {'var':>9s} {'se':>8s} {'theory var':>10s}"]
    for step in ("pm", "uniform"):
        x = simulate_walks(n_max, M, delta, rng, step)
        for n in checkpoints:
            xn = x[:, n]
            mean, var = xn.mean(), xn.var(ddof=1)
            se_mean = xn.std(ddof=1) / np.sqrt(M)
            se_var = var * np.sqrt(2.0 / (M - 1))
            lines.append(f"{step:8s} {n:5d} {mean:8.3f} {se_mean:7.3f} {var:9.2f} {se_var:8.2f} "
                         f"{n * step_variance(delta, step):10.2f}")
    return "\n".join(lines)


def main():
    table = moments_table()
    (HERE / "moments.txt").write_text(table + "\n")
    print(table)
    print("wrote moments.txt")


if __name__ == "__main__":
    main()
