import logging
import warnings

import numpy as np
import pandas as pd

from scipy import integrate, stats
from scipy.interpolate import InterpolatedUnivariateSpline

from .util import log_err_func


def cal_chi_square(f_hat_array, f_array):
    chi_square = np.sum((f_hat_array - f_array) ** 2 / f_hat_array)
    return chi_square


def smoothing(x, y):
    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]
    spl = InterpolatedUnivariateSpline(x, y)
    return spl


def fx_r_from_gy_r(g, p, x, y_min, y_max):
    f_r = np.zeros_like(x, dtype=float)
    for i, x_i in enumerate(x):
        f_r[i] = integrate.quad(lambda y: g(y) * p(x_i, y), y_min, y_max)[0]
    return smoothing(x[f_r > 0], f_r[f_r > 0])


def gy_r1_from_fx_r(f_r, f_hat, g, p, y, x_min, x_max):
    g_r1 = np.zeros_like(y, dtype=float)
    for i, y_i in enumerate(y):
        g_r1[i] = (
            g(y_i)
            * integrate.quad(
                lambda x: f_hat(x) / f_r(x) * p(x, y_i), x_min, x_max, limit=100
            )[0]
        )
    return smoothing(y[g_r1 > 0], g_r1[g_r1 > 0])


def fx_r_from_gy_r_bound(g, p, x, y_min, y_max):
    f_r = np.zeros_like(x, dtype=float)
    for i, x_i in enumerate(x):
        f_r[i] = integrate.quad(
            lambda y: g(y) * p(x_i, y), x_i, y_max, limit=100, points=[x_i]
        )[0]
    return smoothing(x[f_r > 0], f_r[f_r > 0])


def gy_r1_from_fx_r_bound(f_r, f_hat, g, p, y, x_min, x_max):
    g_r1 = np.zeros_like(y, dtype=float)
    for i, y_i in enumerate(y):
        g_r1[i] = (
            g(y_i)
            * integrate.quad(
                lambda x: f_hat(x) / f_r(x) * p(x, y_i),
                x_min,
                y_i,
                limit=100,
                points=[x_min, y_i],
            )[0]
        )
    return smoothing(y[g_r1 > 0], g_r1[g_r1 > 0])


def lucy(p, f_hat, x_array, y_array, limit_i=50, p_ratio=0.97):
    warnings.filterwarnings("ignore")
    g = smoothing(x_array, np.ones_like(x_array) * np.max(f_hat(x_array)) / 2)
    chi_square_p = +np.inf
    i = 0
    for i in range(limit_i):
        f = fx_r_from_gy_r(g, p, x_array, y_min=min(y_array), y_max=max(y_array))
        chi_square = cal_chi_square(f_hat(x_array), f(x_array))
        if chi_square < chi_square_p * p_ratio:
            chi_square_p = chi_square
            g = gy_r1_from_fx_r(
                f, f_hat, g, p, y_array, x_min=min(x_array), x_max=max(x_array)
            )
        else:
            break
    logging.info(f"{i} Lucy iterations")
    return g


def lucy_bound(p, f_hat, x_array, y_array, limit_i=50, p_ratio=0.97):
    warnings.filterwarnings("ignore")
    g = smoothing(x_array, np.ones_like(x_array) * np.max(f_hat(x_array)) / 2)
    chi_square_p = +np.inf
    i = 0
    for i in range(limit_i):
        f = fx_r_from_gy_r_bound(g, p, x_array, y_min=min(y_array), y_max=max(y_array))
        chi_square = cal_chi_square(f_hat(x_array), f(x_array))
        if chi_square < chi_square_p * p_ratio:
            chi_square_p = chi_square
            g = gy_r1_from_fx_r_bound(
                f, f_hat, g, p, y_array, x_min=min(x_array), x_max=max(x_array)
            )
        else:
            break
    logging.info(f"{i} Lucy iterations")
    return g


def cal_v_eq_dist(vsini, popt, only_f=False, limit_i=50):
    s = stats.gaussian_kde(vsini, "silverman")

    f_hat = stats.gaussian_kde(np.log(vsini), "silverman")

    def p(x, y):
        return (
            1
            / (np.sqrt(2 * np.pi) * log_err_func(x, *popt))
            * np.exp(-1 / 2 * ((x - y) / log_err_func(x, *popt)) ** 2)
        )

    x = np.arange(2, 7, 0.05)
    g = lucy(p, f_hat, x, x, limit_i=limit_i)

    x = np.arange(10, 600, 5)
    f_hat = smoothing(x, g(np.log(x)) / x)
    x_output = np.arange(10, 600, 0.5)

    def p(x, y):
        if y > x:
            return x / np.sqrt(y ** 2 - x ** 2) / y
        else:
            return 0

    if not only_f:
        g = lucy_bound(p, f_hat, x, x, limit_i=limit_i)
        df = pd.DataFrame(
            {"x": x_output, "s": s(x_output), "f": f_hat(x_output), "g": g(x_output)}
        )
    else:
        df = pd.DataFrame({"x": x_output, "s": s(x_output), "f": f_hat(x_output)})
    return df


def cal_omega_dist(omega, popt, only_f=False, limit_i=50):
    s = stats.gaussian_kde(omega, "silverman")

    def p(x, y):
        return (
            1
            / (np.sqrt(2 * np.pi) * np.polyval(popt, x))
            * np.exp(-1 / 2 * ((x - y) / np.polyval(popt, x)) ** 2)
        )

    x = np.arange(0, 1.1, 0.02)
    # g = lucy(p, s, x, x, limit_i=limit_i)
    g = s
    f = g

    def p(x, y):
        if y > x:
            return x / np.sqrt(y ** 2 - x ** 2) / y
        else:
            return 0

    if not only_f:
        g = lucy_bound(p, g, x, x, limit_i=limit_i)
        df = pd.DataFrame({"x": x, "s": s(x), "f": f(x), "g": g(x)})
    else:
        df = pd.DataFrame({"x": x, "s": s(x), "f": f(x)})
    return df


if __name__ == "__main__":
    pass
