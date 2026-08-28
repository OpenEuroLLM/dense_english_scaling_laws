import numpy as np

N_palette= [
    "#ff62e0",
    "#ab68d8",
    "#5d63bb",
    "#19548e",
    "#003f5c",
    "#162C36",
]

D_palette = [
    "#10ffbf",
    "#23e4ac",
    "#2bca99",
    "#2fb186",
    "#309874",
    "#308063",
    "#2d6852",
    "#295242",
    "#243c32",
]

FORMAT_BILLIONS = lambda x: f"{float(x)/1e9:g}B"
FORMAT_N = lambda x: f"{float(x)/1e9:.2g}B"
FORMAT_D = lambda x: f"{float(x)/1e9:g}B"

# Approximate param counts
Ns_approx = [
    .05e9, 
    .13e9, 
    .3e9, 
    .6e9, 
    1e9, 
    1.7e9
]
Ns = [
    47_557_632,
    124_455_168,
    301_880_320,
    588_544_000,
    983_138_304,
    1_713_504_256
]
N_to_N_approx = {N: N_approx for N, N_approx in zip(Ns, Ns_approx)}
N_approx_to_N = {N_approx: N for N, N_approx in zip(Ns, Ns_approx)}

## Map N to colors
N_color_map = dict(zip(Ns, N_palette[:len(Ns)]))
N_approx_color_map = dict(zip(Ns_approx, N_palette[:len(Ns_approx)]))

## Map D to colors
Ds = [6e9, 12e9, 20e9, 30e9, 50e9, 80e9, 120e9, 200e9, 300e9]
D_color_map = dict(zip(Ds, D_palette[:len(Ds)]))

# x-ticks for plots
N_ticks = [0.05e9, 0.1e9, 0.2e9, 0.4e9, 0.8e9, 1.6e9]
D_ticks = [5e9, 10e9, 20e9, 40e9, 80e9, 160e9, 320e9,]

def format_sci_latex(a):
    """Return LaTeX string for a in scientific notation (mantissa · 10^{exp})."""
    exp = int(np.floor(np.log10(a)))
    mant = a / 10**exp
    return rf"{mant:.2f} \cdot 10^{{{exp}}}"

SEQ_LEN = 4096