
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

from utils import N_color_map, D_color_map, N_ticks, D_ticks, FORMAT_N, FORMAT_D


def plot_hp_vs_ND_fix_other(
    df_plot,
    y,
    other_fixed,
    savepath=None,
):
    """
    y: "gbsz" or "lr"
    other_fixed: "lr" or "gbsz"
    """

    assert y in ["gbsz", "lr"]
    assert other_fixed in ["gbsz", "lr"]
    assert y != other_fixed

    y_label = {
        'gbsz': r'$b^\star$',
        'lr': r'$\eta\star$',
    }[y]

    xvals = sorted(df_plot[other_fixed].unique())
    n_xvals = len(xvals)

    fig, axes = plt.subplots(
        2,
        n_xvals,
        figsize=(4 * n_xvals, 8),
        squeeze=False,
        sharey="all",
        sharex=False
    )

    N_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=color,
            label=str(FORMAT_N(val))
        )
        for val, color in N_color_map.items()
    ]

    D_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=color,
            label=str(FORMAT_D(val))
        )
        for val, color in D_color_map.items()
    ]

    for c, xval in enumerate(xvals):

        df_b = df_plot[df_plot[other_fixed] == xval]

        # -------------------------------------------------
        # ROW 0: y vs D
        # -------------------------------------------------
        sns.scatterplot(
            data=df_b,
            x="D",
            y=y,
            hue="N",
            palette=N_color_map,
            s=50,
            ax=axes[0, c]
        )

        axes[0, c].set_title(rf"${other_fixed}={xval}$")
        axes[0, c].set_xlabel(r"$D$")
        axes[0, c].set_ylabel(y_label)
        axes[0, c].set_xscale("log")

        if y == "gbsz":
            axes[0, c].set_yscale("log", base=2)
        else:
            axes[0, c].set_yscale("log")

        axes[0, c].legend_.remove()
        axes[0, c].set_xticks(D_ticks)
        axes[0, c].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: FORMAT_D(x))
        )

        # -------------------------------------------------
        # ROW 1: y vs N
        # -------------------------------------------------
        sns.scatterplot(
            data=df_b,
            x="N",
            y=y,
            hue="D",
            palette=D_color_map,
            s=50,
            ax=axes[1, c]
        )

        axes[1, c].set_xlabel(r"$N$")
        axes[1, c].set_ylabel(y_label)
        axes[1, c].set_xscale("log")

        if y == "gbsz":
            axes[1, c].set_yscale("log", base=2)
        else:
            axes[1, c].set_yscale("log")

        axes[1, c].legend_.remove()
        axes[1, c].set_xticks(N_ticks)
        axes[1, c].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: FORMAT_N(x))
        )

    fig.legend(
        handles=N_handles,
        title=r"$N$",
        loc="center left",
        bbox_to_anchor=(0.84, 0.55)
    )

    fig.legend(
        handles=D_handles,
        title=r"$D$",
        loc="center left",
        bbox_to_anchor=(0.84, 0.20)
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=400,
            bbox_inches='tight',
            pad_inches=0.02
        )

    plt.show()


def plot_hp_vs_DN(df_plot, y='lr', savepath=None, N_ticks=N_ticks):

    ylabel = {
        'lr': r'$\eta^\star$',
        'gbsz': r'$b^\star$',
    }[y]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    # D
    sns.scatterplot(
        data=df_plot,
        x="D",
        y=y,
        hue="N",
        palette=N_color_map,
        ax=axes[0],
    )

    axes[0].set_xlabel("D")
    axes[0].set_ylabel(ylabel)

    # N
    sns.scatterplot(
        data=df_plot,
        x="N",
        y=y,
        hue="D",
        palette=D_color_map,
        ax=axes[1],
    )

    axes[1].set_xlabel("N")

    # scales
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)

    # legends
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [FORMAT_N(float(l)) for l in labels]
    axes[0].legend(
        handles,
        labels,
        title=r'$N$',
        fontsize=7,
        ncol=2,
        loc='upper left'
    )

    handles, labels = axes[1].get_legend_handles_labels()
    labels = [FORMAT_D(float(l)) for l in labels]
    axes[1].legend(
        handles,
        labels,
        title=r'$D$',
        fontsize=7,
        ncol=3,
        loc='upper right'
    )

    # ticks
    axes[0].set_xticks(D_ticks)
    axes[0].xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: FORMAT_D(x))
    )

    axes[1].set_xticks(N_ticks)
    axes[1].xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: FORMAT_N(x))
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=400,
            bbox_inches='tight',
            pad_inches=0.02,
        )

    return fig, axes


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_loss_grid(
    df,
    fit,
    y_value="loss",
    lr_range=None,
    bsz_range=None,
    show=True,
    save_path_grid=None,
    save_path_contour=None,
):

    # ------------------------------------------------------------
    # filtering
    # ------------------------------------------------------------

    if lr_range is not None:
        df = df[df["lr"].between(*lr_range)]

    if bsz_range is not None:
        df = df[df["gbsz"].between(*bsz_range)]

    batch_sizes = sorted(df["gbsz"].unique())
    lrs = sorted(df["lr"].unique())

    # ------------------------------------------------------------
    # contour range FIRST
    # ------------------------------------------------------------

    lrs_arr = np.array(lrs)
    bsz_arr = np.array(batch_sizes)

    x_pad = 0.15
    y_pad = 0.15

    x_vals = np.linspace(
        np.log2(lrs_arr.min()) - x_pad,
        np.log2(lrs_arr.max()) + x_pad,
        200,
    )

    y_vals = np.linspace(
        np.log2(bsz_arr.min()) - y_pad,
        np.log2(bsz_arr.max()) + y_pad,
        200,
    )

    X, Y = np.meshgrid(x_vals, y_vals)

    a, b, c, d, e, f_coef = fit.coef

    Z = (
        a
        + b * X
        + c * X**2
        + d * Y
        + e * Y**2
        + f_coef * X * Y
    )

    # SAME RANGE FOR BOTH PLOTS
    vmin = Z.min()
    vmax = Z.max()

    norm = Normalize(vmin=vmin, vmax=vmax)

    # ============================================================
    # heatmap
    # ============================================================

    fig, ax = plt.subplots(figsize=(5, 5))

    pivot = df.pivot_table(
        index="gbsz",
        columns="lr",
        values=y_value,
        aggfunc="min",
    ).reindex(index=batch_sizes, columns=lrs)

    hm = sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        annot_kws={"fontsize": 14},
        fmt=".3f",
        square=True,
        linewidths=1,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": "",
            "shrink": 0.75
        },
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.set_title("Loss", pad=0.04, fontsize=18)

    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=14)
    
    vals = pivot.values

    if not np.isnan(vals).all():
        best_idx = np.nanargmin(vals)
        best_y, best_x = np.unravel_index(best_idx, vals.shape)

        ax.scatter(
            best_x + 0.5,
            best_y + 0.2,
            marker="*",
            s=160,
            c="red",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

    ax.set_xlabel(r"Learning rate $\eta$", fontsize=22)
    ax.set_ylabel(r"Batch size $b$", fontsize=22)
    
    heatmap_ticks = cbar.get_ticks()

    plt.tight_layout()

    if save_path_grid is not None:
        fig.savefig(
            save_path_grid,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ============================================================
    # contour
    # ============================================================

    fig_c, ax = plt.subplots(figsize=(5, 5))

    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=np.linspace(vmin, vmax, 30),
        cmap="viridis",
        norm=norm,
    )

    ax.contour(
        X,
        Y,
        Z,
        levels=np.linspace(vmin, vmax, 15),
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )

    ax.scatter(
        np.log2(df["lr"]),
        np.log2(df["gbsz"]),
        c=df[y_value],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=80,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )

    ax.scatter(
        np.log2(fit.lr_star),
        np.log2(fit.gbsz_star),
        marker="*",
        s=180,
        c="yellow",
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    # empirical best
    vals = pivot.values
    if not np.isnan(vals).all():
        best_idx = np.nanargmin(vals)
        best_y, best_x = np.unravel_index(best_idx, vals.shape)
        best_lr = np.log2(lrs[best_x])
        best_bsz = np.log2(batch_sizes[best_y])
        ax.scatter(
            best_lr,
            best_bsz,
            marker="*",
            s=180,
            c="red",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        

    xticks = np.arange(
        np.floor(x_vals.min()),
        np.ceil(x_vals.max()),
    )

    yticks = np.arange(
        np.floor(y_vals.min()),
        np.ceil(y_vals.max()),
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # ax.set_xticklabels([rf"$2^{{{int(x)}}}$" for x in xticks])
    # ax.set_yticklabels([rf"$2^{{{int(y)}}}$" for y in yticks])
    ax.set_xticklabels([f"{2**x:.4f}".rstrip("0").rstrip(".") for x in xticks])
    ax.set_yticklabels([f"{2**y:.0f}" for y in yticks])

    ax.set_xlabel(r"Learning rate $\eta$", fontsize=22)
    ax.set_ylabel(r"Batch size $b$", fontsize=22)

    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())

    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=14)

    cbar = fig_c.colorbar(
        cf,
        ax=ax,
        shrink=0.7,
        ticks=heatmap_ticks,
    )

    cbar.ax.set_title("Loss", pad=0, fontsize=18)

    plt.tight_layout()

    if save_path_contour is not None:
        fig_c.savefig(
            save_path_contour,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig_c)
