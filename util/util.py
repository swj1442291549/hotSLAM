import glob
import random

from pathlib import Path

import numpy as np
import pandas as pd
import pickle5 as pickle

from astropy.table import Table
from scipy import integrate, stats
from scipy.stats import maxwell
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def select_within_boundary(x, y, boundary):
    polygon = Polygon(boundary)
    p = Point(x, y)
    is_in = polygon.contains(p)
    return is_in


def maxwell_d(x, scale_s, A_s, scale_f, loc_f):
    return A_s * maxwell.pdf(x, scale=scale_s) + (1 - A_s) * maxwell.pdf(
        x, scale=scale_f, loc=loc_f
    )


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def dgauss(x, *p):
    return gauss(x, p[0], p[1], p[2]) + gauss(x, p[3], p[4], p[5])


def sersic(x, a, b, c, d, m):
    return -a * np.exp(-((abs(x - d) / b) ** c)) + m


def pdf(kernel, x, min_x, max_x, int_k):
    """possibility distribution function
    

    Returns:
        p (array-like): probability
    """
    if x < min_x or x > max_x:
        return 0
    else:
        return kernel(x) / int_k


def log_err_func(x, a, b, c):
    return a / (x - b) + c


def gen_kernel_sample(kernel, num, min_x, max_x):
    """Generate mass following Kroupa mass function
    
    Args:
        num (int): number of points
        min_x (float): minimum boundary
        max_x (float): maximum boundary

    Returns:
        result (array): mass
    """
    int_k = integrate.quad(lambda x: kernel(x), min_x, max_x)[0]
    sample = []
    x = np.linspace(min_x, max_x, 100)
    c = pdf(kernel, x[kernel(x).argmax()], min_x, max_x, int_k)
    for i in range(num):
        flag = 0
        while flag == 0:
            x = random.uniform(min_x, max_x)
            y = random.uniform(0, 1)
            if y < pdf(kernel, x, min_x, max_x, int_k) / c:
                sample.append(x)
                flag = 1
    return sample


def z2mh(Z):
    Y = 0.2485 + 1.78 * Z
    X = 1 - Y - Z
    Z_sun = 0.0152
    Y_sun = 0.2485 + 1.78 * Z_sun
    X_sun = 1 - Y_sun - Z_sun
    mh = np.log10(Z / X) - np.log10(Z_sun / X_sun)
    return mh


def get_z_tri(y, dist_dir):
    Z = list()
    dist_list = list()
    for mass in y:
        info = pickle.load(
            open("./pickle/{0}/m{1:.2f}.pkl".format(dist_dir, mass), "rb")
        )
        vsini_dist = info["vsini_dist"]
        vsini_dist_sel = vsini_dist[(vsini_dist.x >= 10) & (vsini_dist.x < 400)]
        vsini_dist_sel = vsini_dist_sel.assign(mass=info["mass_mean"])
        dist_list.append(vsini_dist_sel)
    dist = pd.concat(dist_list)
    X = np.linspace(min(dist.x), max(dist.x), 100)
    Y = np.linspace(min(dist.mass), max(dist.mass), 50)
    Z = griddata((dist.x, dist.mass), dist.g, (X[None, :], Y[:, None]), method="linear")
    Z[Z < 0] = 0
    return X, Y, Z


def get_z_tri_dist(y, dist_dir):
    dist_list = list()
    for mass in y:
        info = pickle.load(
            open("./pickle/{0}/m{1:.2f}.pkl".format(dist_dir, mass), "rb")
        )
        vsini_dist = info["vsini_dist"]
        vsini_dist_sel = vsini_dist[(vsini_dist.x >= 20) & (vsini_dist.x < 400)]
        vsini_dist_sel = vsini_dist_sel.assign(mass=info["mass_mean"])
        dist_list.append(vsini_dist_sel)
    dist = pd.concat(dist_list)
    return dist


def get_z_omega_dist(y, dist_dir, ratio):
    df = Table.read("./material/Netopil2017.fit").to_pandas()
    df = df[df.v_vcrit > 0]

    dist_list = list()
    for mass in y:
        info = pickle.load(
            open("./pickle/{0}/m{1:.2f}.pkl".format(dist_dir, mass), "rb")
        )
        omega_dist = info["omega_dist"]
        omega_dist.g[omega_dist.x > 1.03] = 0
        g_new = clean_omega_cp_dist(omega_dist, df.v_vcrit, ratio)
        omega_dist = omega_dist.assign(mass=info["mass_mean"])
        omega_dist.g = g_new
        dist_list.append(omega_dist)
    dist = pd.concat(dist_list)
    return dist


def get_mass_lower_dir(dist_dir, mass_l=1.7, mass_u=3.7):
    mass_lower_list = list()
    for file_name in glob.glob("./pickle/{0}/*.pkl".format(dist_dir)):
        mass_lower_list.append(float(Path(file_name).stem[1:]))
    mass_lower_array = np.sort(mass_lower_list)
    mass_lower_array = mass_lower_array[
        (mass_lower_array > mass_l) & (mass_lower_array < mass_u)
    ]
    return mass_lower_array


def clean_omega_cp_dist(omega_dist, cp_omega, ratio):
    s_cp = stats.gaussian_kde(cp_omega, "silverman")
    g_new = omega_dist.g - s_cp(omega_dist.x) * ratio
    g_new[g_new < 0] = 0
    delta_x = omega_dist.x[1] - omega_dist.x[0]
    g_new = g_new / np.sum(g_new) / delta_x
    return g_new


if __name__ == "__main__":
    pass
