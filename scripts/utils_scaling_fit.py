"""
Helper functions to fit quadratic functions of lr, bsz, or both.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import N_color_map, D_color_map, FORMAT_N, FORMAT_D, N_ticks, D_ticks


def plot_diagnostic(fit):

    y_true = fit.model.endog

    # fitted + residuals
    y_pred = fit.fittedvalues
    infl = fit.get_influence()
    resid_std = infl.resid_studentized_internal

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # --- (1) Residuals vs Fitted ---
    axes[0].scatter(y_pred, resid_std)
    axes[0].axhline(0)
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Studentized residuals")
    axes[0].set_title("Residuals vs Fitted")

    # --- (2) QQ plot ---
    sm.qqplot(resid_std, line='45', fit=True, ax=axes[1])
    axes[1].set_title("QQ plot")

    # --- (3) y vs y_hat ---
    axes[2].scatter(y_true, y_pred)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    axes[2].plot(lims, lims)
    axes[2].set_xlabel("Observed $y$")
    axes[2].set_ylabel("Fitted $\hat{y}$")
    axes[2].set_title(r"$y$ vs $\hat{y}$")

    plt.tight_layout()
    plt.show()



def plot_diagnostic_1d(fit, y="y"):
    """Plot Linear Model fit vs x, together with residuals and QQ-plot"""
    X = fit.model.exog
    y_data = fit.model.endog

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    # y ~ x + fitted line
    xi = X.iloc[:, 1]
    ax[0].scatter(xi, y_data)

    # fitted line (smooth)
    x_grid = np.linspace(xi.min(), xi.max(), 100)
    X_line = X.iloc[[0]*100].copy()
    X_line.iloc[:, 1] = x_grid

    y_line = fit.predict(X_line)
    ax[0].plot(x_grid, y_line)

    ax[0].set_title(f"{y} vs x")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel(y)

    # Residuals vs Fitted
    ax[1].scatter(fit.fittedvalues, fit.resid)
    ax[1].axhline(0, linestyle="--")
    ax[1].set_title("Residuals vs Fitted")

    # Q-Q
    qqplot(fit.resid, line="45", ax=ax[2])
    ax[2].set_title("Normal Q-Q")

    plt.tight_layout()
    plt.show()


def plot_diagnostic_2d(fit, y="y", x=None):

    X = fit.model.exog
    y_data = fit.model.endog

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if x is None:
        x = X.columns[1:]

    x1 = X.iloc[:, 1]
    x2 = X.iloc[:, 2]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            f"{y} vs {x[0]}, {x[1]}",
            "Residuals vs Fitted",
            "Normal Q-Q"
        )
    )

    # --- 1) 3D ---
    fig.add_trace(
        go.Scatter3d(x=x1, y=x2, z=y_data, mode="markers"),
        row=1, col=1
    )

    x1g = np.linspace(x1.min(), x1.max(), 30)
    x2g = np.linspace(x2.min(), x2.max(), 30)
    X1g, X2g = np.meshgrid(x1g, x2g)

    X_grid = pd.DataFrame({
        X.columns[0]: 1,
        X.columns[1]: X1g.ravel(),
        X.columns[2]: X2g.ravel()
    })

    Z = fit.predict(X_grid).values.reshape(X1g.shape)

    fig.add_trace(
        go.Surface(x=X1g, y=X2g, z=Z, showscale=False, opacity=0.5),
        row=1, col=1
    )

    fig.update_scenes(
        xaxis_title=str(x[0]),
        yaxis_title=str(x[1]),
        zaxis_title=y,
        row=1, col=1
    )

    # --- 2) residuals ---
    fig.add_trace(
        go.Scatter(x=fit.fittedvalues, y=fit.resid, mode="markers"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=fit.fittedvalues,
            y=[0]*len(fit.fittedvalues),
            mode="lines",
            line=dict(dash="dash")
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Fitted values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)

    # --- 3) QQ ---
    qq = stats.probplot(fit.resid, dist="norm")
    theo, sample = qq[0]

    fig.add_trace(go.Scatter(x=theo, y=sample, mode="markers"), row=1, col=3)
    fig.add_trace(go.Scatter(x=theo, y=theo, mode="lines"), row=1, col=3)

    fig.update_xaxes(title_text="Theoretical quantiles", row=1, col=3)
    fig.update_yaxes(title_text="Sample quantiles", row=1, col=3)

    fig.update_layout(height=400, width=1200, showlegend=False)
    fig.show()


def bootstrap_test_null(df, y, cols_full, cols_restricted, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    
    # fit restricted model
    Xr = sm.add_constant(df[cols_restricted])
    mr = sm.OLS(df[y], Xr).fit()
    
    yr_hat = mr.fittedvalues
    ur = mr.resid.values
    n = len(df)
    
    # observed stat
    def rss(cols, y):
        X = sm.add_constant(df[cols])
        return np.sum(sm.OLS(y, X).fit().resid**2)
    
    T_obs = rss(cols_restricted, df[y]) - rss(cols_full, df[y])
    
    T_boot = []
    
    for _ in range(B):
        u_star = rng.choice(ur, size=n, replace=True)
        y_star = yr_hat + u_star
        
        T_b = rss(cols_restricted, y_star) - rss(cols_full, y_star)
        T_boot.append(T_b)
    
    T_boot = np.array(T_boot)
    
    pval = np.mean(T_boot >= T_obs)
    
    return T_obs, pval
