"""
Helper functions to fit quadratic functions of lr, bsz, or both.
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

from utils import FORMAT_N, FORMAT_D


def fit_2d_quadratic(df, method='ols', y='loss', x='lr', z='gbsz'):
    """
    Fit quadratic model of validation loss vs log(lr) and log(gbsz):

        loss(lr,gbsz | N,D) ≈ a_{N,D} + b_{N,D} log(lr) 
                            + c_{N,D} log(lr)^2
                            + d_{N,D} log(gbsz) 
                            + e log(gbsz)^2
                            + f_{N,D} log(lr)log(gbsz)
    """
    x = np.log2(df[x].values)
    z = np.log2(df[z].values)
    y = df[y].values

    X = np.column_stack([
        np.ones_like(x),  # a -> explicit intercept
        x,                # b
        x**2,             # c
        z,                # d
        z**2,             # e
        x*z               # f
    ])

    if method == 'ols':
        model = sm.OLS(y, X)
    elif method == 'HuberT':
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    elif method == 'TukeyBiweight':
        model = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())

    return model.fit()


def minima_2d_quadratic(coef):
    """
    Find LR and GBSZ that minimize the quadratic surface
    by solving ∂loss/∂log(lr)=0 and ∂loss/∂log(gbsz)=0.
    
    Returns:
        (lr_star, gbsz_star): learning rate and batch size that minimize
        the fitted quadratic loss surface.
    """
    a,b,c,d,e,f = coef

    # gradient = 0 -> linear system
    # b + 2c x + f z = 0
    # d + 2e z + f x = 0
    A = np.array([[2*c, f],
                  [f, 2*e]])
    rhs = np.array([-b, -d])

    x_opt, z_opt = np.linalg.solve(A, rhs)
    
    # value at minimum
    loss_min = (
        a
        + b*x_opt
        + c*x_opt**2
        + d*z_opt
        + e*z_opt**2
        + f*x_opt*z_opt
    )

    return 2**x_opt, 2**z_opt, loss_min   # lr*, gbsz*, loss(lr*, gbsz*)


def plot_2d_quadratic(results, df, save_path=None):
    """
    Plot fitted quadratic surfaces from `results`.

    Args:
        results (pd.DataFrame): must contain columns
            - N, D
            - coef: iterable [a,b,c,d,e,f]
            - lr_star, gbsz_star
        df (pd.DataFrame): raw runs with columns
            - N, D, lr, gbsz, lm_loss_valid
    """

    df_plot = results.copy()
    Ns = sorted(df_plot["N"].unique())
    Ds = sorted(df_plot["D"].unique())

    fig, axes = plt.subplots(
        len(Ns), len(Ds),
        figsize=(4 * len(Ds), 3 * len(Ns)),
        squeeze=False,
        sharex=True,
        sharey=True
    )

    for i, N in enumerate(Ns):
        for j, D in enumerate(Ds):
            ax = axes[i, j]

            row = df_plot[(df_plot.N == N) & (df_plot.D == D)]
            if row.empty:
                ax.axis("off")
                continue

            r = row.iloc[0]

            a, b, c, d, e, f = r["coef"]

            lr_star, gbsz_star = r["lr_star"], r["gbsz_star"]

            # raw data for this (N,D)
            g = df[(df.N == N) & (df.D == D)]
            if g.empty:
                ax.axis("off")
                continue

            lr = np.geomspace(g.lr.min() / 2, g.lr.max() * 2, 60)
            bsz = np.geomspace(g.gbsz.min() / 2, g.gbsz.max() * 2, 60)

            X, Z = np.meshgrid(np.log2(lr), np.log2(bsz))
            Y = a + b*X + c*X**2 + d*Z + e*Z**2 + f*X*Z

            ax.contourf(lr, bsz, Y, levels=20) # fitted surface
            ax.scatter(g.lr, g.gbsz, c="red", s=10) # observations

            # best empirical
            best_row = g.loc[g['loss'].idxmin()]
            ax.scatter(
                best_row.lr, best_row.gbsz,
                marker="*", s=180, color="yellow",
                edgecolor="black", zorder=10
            )

            # best predicted
            if (lr.min() <= lr_star <= lr.max()) and (bsz.min() <= gbsz_star <= bsz.max()):
                ax.scatter(
                    lr_star, gbsz_star,
                    marker="*", s=180, color="limegreen",
                    edgecolor="black", zorder=11
                )

            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)

            if i == 0:
                ax.set_title(f"{FORMAT_D(D)}T")
            if i == len(Ns) - 1:
                ax.set_xlabel(r"Learning rate $\eta$")
            if j == 0:
                ax.set_ylabel(r"Batch size $b$")

    # row labels for N
    for i, N in enumerate(Ns):
        axes[i, 0].annotate(
            f"N={FORMAT_N(N)}",
            xy=(-0.2, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            weight="bold"
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.show()



def plot_3d_quadratic_surface(data, fit, save_path=None):
    """A 3D plotly plot."""

    # grid (log2 space)
    lrs = np.sort(data['lr'].unique())
    batch_sizes = np.sort(data['gbsz'].unique())

    x_vals = np.linspace(np.log2(lrs.min()), np.log2(lrs.max()), 60)
    z_vals = np.linspace(np.log2(batch_sizes.min()), np.log2(batch_sizes.max()), 60)
    X, Z = np.meshgrid(x_vals, z_vals)

    # coefficients
    a, b, c, d, e, f = fit.coef

    # surface
    L = a + b*X + c*X**2 + d*Z + e*Z**2 + f*X*Z

    fig = go.Figure()

    # --- surface ---
    fig.add_trace(go.Surface(
        x=X,
        y=Z,
        z=L,
        colorscale="viridis",
        opacity=0.8,
        showscale=False,
        name=""
    ))

    # --- data points ---
    fig.add_trace(go.Scatter3d(
        x=np.log2(data['lr']),
        y=np.log2(data['gbsz']),
        z=data['loss'],
        mode='markers',
        marker=dict(size=6, color="blue"),
        name=""
    ))

    # --- optimum ---
    fig.add_trace(go.Scatter3d(
        x=[np.log2(fit.lr_star)],
        y=[np.log2(fit.gbsz_star)],
        z=[fit.loss_star],
        mode='markers',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        name="",
    ))
    
    # # --- slices + optimal LR for each observed batch size ---

    # bsz_obs = np.sort(data["gbsz"].unique())
    # y_obs = np.log2(bsz_obs)

    # x_line = np.linspace(x_vals.min(), x_vals.max(), 200)

    # x_star = []
    # l_star = []

    # for y in y_obs:

    #     # quadratic slice at fixed batch size
    #     l_line = (
    #         a
    #         + b * x_line
    #         + c * x_line**2
    #         + d * y
    #         + e * y**2
    #         + f * x_line * y
    #     )

    #     fig.add_trace(go.Scatter3d(
    #         x=x_line,
    #         y=np.full_like(x_line, y),
    #         z=l_line,
    #         mode="lines",
    #         line=dict(color="limegreen", width=4),
    #         showlegend=False,
    #     ))

    #     # optimum along this slice
    #     xs = -(b + f * y) / (2 * c)
    #     ls = (
    #         a
    #         + b * xs
    #         + c * xs**2
    #         + d * y
    #         + e * y**2
    #         + f * xs * y
    #     )

    #     x_star.append(xs)
    #     l_star.append(ls)

    # fig.add_trace(go.Scatter3d(
    #     x=x_star,
    #     y=y_obs,
    #     z=l_star,
    #     mode="markers",
    #     marker=dict(
    #         size=7,
    #         color="limegreen",
    #         symbol="diamond",
    #     ),
    #     showlegend=False,
    # ))
    

    # ticks as 2^k
    def ticks(vals):
        t = np.arange(np.floor(vals.min()), np.ceil(vals.max()) + 1)
        return t, [f"2^{int(v)}" for v in t]

    xticks, xlabels = ticks(x_vals)
    yticks, ylabels = ticks(z_vals)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Learning Rate",
                title_font=dict(size=30),
                tickvals=xticks,
                ticktext=xlabels,
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title="Batch Size",
                title_font=dict(size=30),
                tickvals=yticks,
                ticktext=ylabels,
                tickfont=dict(size=14),
            ),
            zaxis=dict(
                title="Loss",
                title_font=dict(size=30),
                tickfont=dict(size=14),
            ),
        ),
        height=700,
        showlegend=False,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  # kill outer whitespace
    )

    fig.update_layout(
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),  # use full canvas

            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),  # compress vertical slack

            camera=dict(
                eye=dict(x=1.1, y=1.1, z=0.7)  # closer = less empty space
            )
        )
    )

    fig.show()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path, width=1200, height=800, scale=2)
