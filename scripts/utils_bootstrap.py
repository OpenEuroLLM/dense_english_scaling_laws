import numpy as np
from scipy.optimize import minimize
from scipy.special import huber


def huber_objective(params, xdata, ydata, scaling_law, delta):
    """
    Huber loss on log-space residuals:
        r_i = log(L_true_i) - log(L_pred_i)
    """
    ypred = scaling_law(xdata, *params)

    if np.any(ypred <= 0) or np.any(ydata <= 0):
        return np.inf

    residuals = np.log(ydata) - np.log(ypred)

    return np.sum(huber(delta, residuals))


def fit_initialization_chunk(
    x_boot,
    y_boot,
    initializations,
    scaling_law,
    huber_delta,
    minimize_options,
):
    """
    Run L-BFGS-B from a chunk of initializations for ONE bootstrap sample.

    Returns the best (params, objective) found in this chunk.
    """

    best_params = None
    best_objective = np.inf

    for p0 in initializations:

        result = minimize(
            huber_objective,
            x0=np.asarray(p0, dtype=float),
            args=(x_boot, y_boot, scaling_law, huber_delta),
            method="L-BFGS-B",
            bounds=None,
            options=minimize_options,
        )

        if (
            result.success
            and np.isfinite(result.fun)
            and result.fun < best_objective
        ):
            best_objective = result.fun
            best_params = result.x

    return best_params, best_objective


def one_bootstrap_init_chunk(
    bootstrap_id,
    seed,
    xdata,
    ydata,
    initializations,
    scaling_law,
    huber_delta,
    minimize_options,
):
    """
    One parallel task.

    Generates the bootstrap sample for `bootstrap_id`, then
    fits ONE CHUNK of initializations.
    """

    rng = np.random.default_rng(seed)

    # Bootstrap resample
    idx = rng.integers(
        0,
        len(ydata),
        size=len(ydata),
    )

    x_boot = xdata[:, idx]
    y_boot = ydata[idx]

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        params, objective = fit_initialization_chunk(
            x_boot=x_boot,
            y_boot=y_boot,
            initializations=initializations,
            scaling_law=scaling_law,
            huber_delta=huber_delta,
            minimize_options=minimize_options,
        )

    return bootstrap_id, params, objective