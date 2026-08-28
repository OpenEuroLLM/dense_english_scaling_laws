
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from utils_bootstrap import one_bootstrap_init_chunk

N_JOBS = 4
N_BOOTSTRAP = 1_000
N_INIT_PER_CHUNK = 100


def scaling_law_skaling(X, log_E, log_A, alpha, log_B, beta, k):
    log_N, log_D = X

    E = np.exp(log_E)
    A = np.exp(log_A)
    B = np.exp(log_B)

    return E + (
        A * np.exp(-alpha * log_N) + B * np.exp(-beta * log_D)
    ) ** k


def main():

    # -----------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------

    loss_key = 'loss'

    # Optimal (lr, gbsz) for each (N, D)
    df_best_lr_gbsz = pd.read_csv('../data//best_HPs/lr_gbsz_min.csv')

    # Reserve held-out 1.7B set and fit
    df_base = df_best_lr_gbsz.copy()
    df_train = df_base[df_base.N_approx < 1.7e9].copy()

    df_fit = df_train.copy()

    xdata = np.vstack([df_fit["N"].values, df_fit["D"].values])
    xdata_log = np.log(xdata)
    ydata = df_fit[loss_key].values

    # -----------------------------------------------------------------------------
    # Settings and Scaling Law
    # -----------------------------------------------------------------------------

    HUBER_DELTA = 1e-3

    minimize_options={
        "maxiter": 100_000,
        "ftol": 1e-12,
        "gtol": 1e-8,
    }

    # ------------------------------------------------------------
    # Deterministic initialization grid
    # ------------------------------------------------------------

    # Hoffman grid
    E_log_grid  = [-1., -0.5, 0., 0.5, 1.]
    log_A_grid  = [0., 20.]
    log_B_grid  = [0., 20.]
    alpha_grid  = [0., 0.5, 1., 1.5]
    beta_grid   = [0., 0.5, 1., 1.5]
    kappa_grid  = [0., 0.5, 1., 1.5]

    initializations = list(itertools.product(
        E_log_grid,
        log_A_grid,
        alpha_grid,
        log_B_grid,
        beta_grid,
        kappa_grid
    ))

    ninit = len(initializations)

    print(f"Number of initializations: {ninit}")

    # -------------------------------------------------------------------------
    # Split initialization grid into chunks
    # -------------------------------------------------------------------------

    init_chunks = [
        initializations[i:i + N_INIT_PER_CHUNK]
        for i in range(0, ninit, N_INIT_PER_CHUNK)
    ]

    n_chunks = len(init_chunks)

    print(f"Initializations per chunk: {N_INIT_PER_CHUNK}")
    print(f"Chunks per bootstrap: {n_chunks}")
    print(f"Total parallel tasks: {N_BOOTSTRAP * n_chunks}")

    # -------------------------------------------------------------------------
    # Bootstrap seeds
    # -------------------------------------------------------------------------

    seed_sequences = np.random.SeedSequence(1996).spawn(N_BOOTSTRAP)

    # Convert to plain integers so they serialize cheaply.
    seeds = [
        int(seed.generate_state(1)[0])
        for seed in seed_sequences
    ]

    # -------------------------------------------------------------------------
    # Create parallel tasks
    # -------------------------------------------------------------------------

    tasks = (
        (
            bootstrap_id,
            seeds[bootstrap_id],
            init_chunk,
        )
        for bootstrap_id in range(N_BOOTSTRAP)
        for init_chunk in init_chunks
    )

    # -------------------------------------------------------------------------
    # Run everything
    # -------------------------------------------------------------------------

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(one_bootstrap_init_chunk)(
            bootstrap_id=bootstrap_id,
            seed=seed,
            xdata=xdata_log,
            ydata=ydata,
            initializations=init_chunk,
            scaling_law=scaling_law_skaling,
            huber_delta=HUBER_DELTA,
            minimize_options=minimize_options,
        )
        for bootstrap_id, seed, init_chunk in tasks
    )

    # -------------------------------------------------------------------------
    # Reduce: take the minimum across initialization chunks
    # -------------------------------------------------------------------------

    params_bootstrap = np.empty(
        (N_BOOTSTRAP, 6),
        dtype=float,
    )

    objective_bootstrap = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for bootstrap_id in range(N_BOOTSTRAP):

        bootstrap_results = [
            result
            for result in results
            if result[0] == bootstrap_id
        ]

        best_params, best_objective = min(
            bootstrap_results,
            key=lambda x: x[2],
        )[1:]

        params_bootstrap[bootstrap_id] = best_params
        objective_bootstrap[bootstrap_id] = best_objective

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    np.savez(
        "../data/bootstrap/bootstrap_skalling.npz",
        params_bootstrap=params_bootstrap,
        objective_bootstrap=objective_bootstrap,
        seeds=np.array(seeds),
    )

    print(f"Saved {N_BOOTSTRAP} bootstrap fits.")


if __name__ == "__main__":
    main()