import numpy as np

wave_array = np.append(np.arange(4990, 5310, 0.05), np.arange(6340, 6810, 0.05))
wave_mrs_b_array = np.arange(5000, 5300, 0.1)
wave_mrs_r_array = np.arange(6350, 6800, 0.1)
wave_mrs_array = np.append(wave_mrs_b_array, wave_mrs_r_array)
wave_range_list = [[5000, 5300], [6350, 6800]]

log_err_popt = [2.9997, -1.12222, -0.3896]

omega_popt = [0.08233346, 0.01759387]
