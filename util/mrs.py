import warnings

from pathlib import Path

import pandas as pd

from scipy.signal import correlate
from scipy.optimize import curve_fit
from laspec.mrs import MrsSpec
from laspec.wavelength import vac2air
from astropy import constants
from astropy.io import fits

from .util import dgauss, gauss, sersic
from .values import *

warnings.simplefilter(action="ignore", category=RuntimeWarning)


def cal_rv(ms, wavelength=6562.80277):
    try:
        popt, pcov = curve_fit(
            sersic, ms.wave, ms.flux_norm, p0=(1, 1, 1, wavelength, 1)
        )
    except:
        return np.nan, np.nan
    perr = np.sqrt(np.diag(pcov))
    rv = (popt[3] / wavelength - 1) * constants.c.cgs.value / 1e5
    rv_err = (perr[3] / wavelength) * constants.c.cgs.value / 1e5
    return rv, rv_err


def combine_band(ms_b, ms_r):
    ms_b.wave = np.append(ms_b.wave, ms_r.wave)
    ms_b.flux = np.append(ms_b.flux, ms_r.flux)
    ms_b.flux_cont = np.append(ms_b.flux_cont, ms_r.flux_cont)
    ms_b.flux_err = np.append(ms_b.flux_err, ms_r.flux_err)
    ms_b.flux_norm = np.append(ms_b.flux_norm, ms_r.flux_norm)
    ms_b.flux_norm_err = np.append(ms_b.flux_norm_err, ms_r.flux_norm_err)
    ms_b.ivar = np.append(ms_b.ivar, ms_r.ivar)
    ms_b.ivar_norm = np.append(ms_b.ivar_norm, ms_r.ivar_norm)
    ms_b.mask = np.append(ms_b.mask, ms_r.mask)
    return ms_b


def read_mrs_spectrum(file_name, rvb, rvr, **kargs):
    ms_r = read_mrs_band_rv_spectrum(file_name, "COADD_R", **kargs)
    ms_b = read_mrs_band_rv_spectrum(file_name, "COADD_B", **kargs)
    ms_r.wave = ms_r.wave_rv(rvr)
    ms_b.wave = ms_b.wave_rv(rvb)
    return combine_band(ms_b, ms_r)


def read_mrs_band_rv_spectrum(file_name, ext, **kargs):
    ms = MrsSpec.from_mrs(file_name, ext, **kargs)
    ms.wave = vac2air(ms.wave)
    return ms


def read_mrs_single_spectrum(file_name, rv, ext_r, ext_b, **kargs):
    ms_r = read_mrs_band_rv_spectrum(file_name, ext_r, **kargs)
    ms_b = read_mrs_band_rv_spectrum(file_name, ext_b, **kargs)
    ms_r.wave = ms_r.wave_rv(rv)
    ms_b.wave = ms_b.wave_rv(rv)
    return combine_band(ms_b, ms_r)


def prepare_mrs_data(ms_list, wave):
    flux_norm = np.array(
        [
            np.interp(wave, ms.wave[ms.mask == 0], ms.flux_norm[ms.mask == 0])
            for ms in ms_list
        ]
    )
    ivar_norm = np.array(
        [
            np.interp(wave, ms.wave[ms.mask == 0], ms.flux_norm_err[ms.mask == 0] ** -2)
            for ms in ms_list
        ]
    )
    return flux_norm, ivar_norm


def prepare_mrs_ori_data(ms_list, wave):
    flux = np.array(
        [
            np.interp(wave, ms.wave[ms.mask == 0], ms.flux[ms.mask == 0])
            for ms in ms_list
        ]
    )
    ivar = np.array(
        [
            np.interp(wave, ms.wave[ms.mask == 0], ms.flux_err[ms.mask == 0] ** -2)
            for ms in ms_list
        ]
    )
    return flux, ivar


def extract_mrs_info(file_name, rvb, rvr, **kargs):
    ms = read_mrs_spectrum(file_name, rvb, rvr, **kargs)
    flux_norm, ivar_norm = prepare_mrs_data([ms], wave_mrs_array)

    ms_n = pd.DataFrame({"wave": wave_mrs_array, "flux": flux_norm[0]})

    return {
        "lw_he": cal_line_width(ms_n, [6672, 6689], 0.1),
        "lw_h": cal_line_width(ms_n, [6548, 6578], 0.1),
        "lw_mg": cal_line_width(ms_n, [5160.125, 5192.625], 0.1),
    }


def extract_mrs_ori_info(file_name, rv, **kargs):
    ms = read_mrs_spectrum(file_name, rv, **kargs)
    flux, ivar = prepare_mrs_ori_data([ms], wave_mrs_array)

    ms_n = pd.DataFrame({"wave": wave_mrs_array, "flux": flux[0]})

    return {
        "lw_he": cal_equivalent_width(
            ms_n, [6672, 6689], [[6653, 6672], [6690, 6712]], 0.1
        ),
        "lw_h": cal_equivalent_width(
            ms_n, [6548, 6578], [[6420, 6455], [6600, 6640]], 0.1
        ),
        "lw_mg": cal_equivalent_width(
            ms_n,
            [5160.125, 5192.625],
            [[5142.625, 5161.375], [5191.375, 5206.375]],
            0.1,
        ),
    }


def extract_mrs_drv(file_name, **kargs):
    snr_list = list()
    ext_list = list()
    hdu_list = fits.open(file_name)
    for j in range(len(hdu_list)):
        if hdu_list[j].header["EXTNAME"].startswith("R-"):
            ext_list.append(hdu_list[j].header["EXTNAME"])
            snr_list.append(hdu_list[j].header["SNR"])

    index = np.array(snr_list).argmax()

    ext_ref = ext_list[index]
    ms_ref = read_mrs_band_rv_spectrum(file_name, ext_ref, **kargs)

    try:
        wave_array = np.arange(6350, 6800, 0.01)
        flux_norm_ref, ivar_norm_ref = prepare_mrs_data([ms_ref], wave_array)
        flux_norm_ref[flux_norm_ref < 0] = 0
        flux_norm_ref[flux_norm_ref > 1.5] = 1
    except:
        return {"drv": np.array([]), "drv_err": np.array([])}

    drv = list()
    drv_err = list()
    drv.append(0)
    drv_err.append(0)
    for ext in ext_list:
        if ext is not ext_ref:
            ms = read_mrs_band_rv_spectrum(file_name, ext, **kargs)
            if ms.snr > 3:
                flux_norm, ivar_norm = prepare_mrs_data([ms], wave_array)
                flux_norm[flux_norm < 0] = 0
                flux_norm[flux_norm > 1.5] = 1
                corr = correlate(1 - flux_norm_ref[0], 1 - flux_norm[0], "same")
                x = wave_array[(wave_array > 6560) & (wave_array < 6590)]
                y = corr[(wave_array > 6560) & (wave_array < 6590)]
                try:
                    popt, pcov = curve_fit(gauss, x, y, p0=(20, 6575, 5))
                    perr = np.sqrt(np.diag(pcov))
                    dwave = popt[1]
                    dwave_err = perr[1]
                    if np.abs(popt[2]) < 0.7:
                        popt, pcov = curve_fit(
                            dgauss, x, y, p0=(20, 6575, 5, 20, 6576, 0.5)
                        )
                        perr = np.sqrt(np.diag(pcov))
                        popt = popt.reshape(2, 3)
                        perr = perr.reshape(2, 3)
                        dwave = popt[popt[:, 2] > 0.7][0][1]
                        dwave_err = perr[popt[:, 2] > 0.7][0][1]
                except:
                    continue
                drv.append((dwave - np.mean(wave_array)) / np.mean(wave_array) * 3e5)
                drv_err.append(dwave_err / np.mean(wave_array) * 3e5)
    return {"drv": np.array(drv), "drv_err": np.array(drv_err)}


def check_filepath(df, folder_path):
    default_status = True
    num = 0
    for i in range(len(df)):
        if not Path(folder_path, df.iloc[i]["stem"]).is_file():
            print("{0} not found!".format(df.iloc[i]["stem"]))
            default_status = False
            num += 1
    if default_status:
        print("all files found")
    else:
        print("{0:d} files not found".format(num))


def cal_line_width(data, wave_range, line_step):
    data_sel = data[(data.wave > wave_range[0]) & (data.wave < wave_range[1])]
    return (len(data_sel) - np.sum(data_sel.flux)) * line_step


def cal_equivalent_width(data, band_wave_range, pseudo_wave_range_list, line_step):
    d = list()
    for wave_range in pseudo_wave_range_list:
        data_sel = data[(data.wave > wave_range[0]) & (data.wave < wave_range[1])]
        d.append(data_sel)
    d = pd.concat(d)
    popt = np.polyfit(d.wave, d.flux, 1)
    data_sel = data[(data.wave > band_wave_range[0]) & (data.wave < band_wave_range[1])]
    return np.sum(1 - data_sel.flux / np.polyval(popt, data_sel.wave)) * line_step
