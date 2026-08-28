"""
Helper functions to fit quadratic functions of lr, bsz, or both.
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def fit_1d_quadratic(df, loss_col='loss', x_col='lr'):
    """
    Fit quadratic model of validation loss on log(x):
        loss(lr, | N,D,B) ~ a_{N,D,B} + b_{N,D,B} log(x) + c_{N,D,B} log(x)^2
    """
    x = np.log2(df[x_col].values)
    y = df[loss_col].values

    X = np.column_stack([
        np.ones_like(x),  # a -> explicit intercept
        x,                # b
        x**2,             # c
    ])

    model = sm.OLS(y, X)

    return model.fit()

def minima_1d_quadratic(coef):
    """
    Returns:
        lr_star: learning rate that minimizes the fitted quadratic loss surface.
        loss_min: loss value at lr_star
    """
    a,b,c = coef

    if c <= 0:
        return None, None

    x_opt = -b / (2 * c)
    loss_min = a - (b ** 2) / (4 * c)

    return x_opt, loss_min


def slice_lr(coef, bsz):
    """
    L(lr,gbsz) ~ a + b log(lr) + c log(lr)^2
                   + d log(gbsz) + e log(gbsz)^2
                   + f log(lr)log(gbsz)

    Returns (a,b,c) for L(x | z=log2(bsz)) = a + b x + c x^2
    """
    c0, c1, c2, c3, c4, c5 = coef
    z0 = np.log2(bsz)

    a = c0 + c3*z0 + c4*z0**2
    b = c1 + c5*z0
    c = c2

    return a, b, c


def slice_bsz(coef, lr):
    """
    L(lr,gbsz) ~ a + b log(lr) + c log(lr)^2
                   + d log(gbsz) + e log(gbsz)^2
                   + f log(lr)log(gbsz)

    Returns (a,b,c) for L(z | x=log2(lr)) = a + b z + c z^2
    """
    c0, c1, c2, c3, c4, c5 = coef
    x0 = np.log2(lr)

    a = c0 + c1*x0 + c2*x0**2
    b = c3 + c5*x0
    c = c4

    return a, b, c
