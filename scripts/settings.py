"""
Common settings for analysis and plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

sns.set_theme(
  style="whitegrid",
  rc={
        "grid.color": "gray",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
  }
)
# Font sizes
FONT_SIZE = 8
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.5
TITLE_SIZE = 11

tex_fonts = {
    "text.usetex": True, # Use LaTeX to write all text
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    
    "font.size": FONT_SIZE,
    
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,

    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,

    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    
    # "axes.titlesize": 12,  # Title font size
    # "axes.labelsize": 10,  # Axis label font size
    # "font.size": 10,       # General font size
    # "legend.fontsize": 10, # Legend font size
    # "legend.title_fontsize": 10,  # Legend title font size
    # "xtick.labelsize": 10, # X-tick label size
    # "ytick.labelsize": 10, # Y-tick label size
}

plt.rcParams.update(tex_fonts)
plt.rcParams['axes.facecolor'] = '#fcfcfc'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 400
plt.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams.update({
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    # "axes.edgecolor": "#505050",
    "axes.edgecolor": "#7B7B7B",
    "axes.linewidth": 0.5,
})
