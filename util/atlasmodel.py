import pickle

from pathlib import Path

import numpy as np
import pandas as pd

from pysynphot import observation, spectrum
from PyAstronomy import pyasl
from scipy.signal import convolve
from astropy import constants
from astropy.convolution import Gaussian1DKernel


class Model:
    def __init__(self, T, logg, z, norm=False, data=None):
        self.T = int(np.round(T / 100)) * 100
        self.logg = np.round(logg / 0.1) * 0.1
        self.z = np.round(z / 0.5) * 0.5
        if self.z == 0.0:
            self.z = 0.0
        self.filename = "A_p{0}g{1:.1f}z{2:.1f}t2.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec".format(
            self.T, self.logg, self.z
        )
        if data is None:
            if not norm:
                pkl_file = Path(
                    "model/pickle/T{0}g{1:.1f}z{2:0.1f}.pkl".format(
                        self.T, self.logg, self.z
                    )
                )
                if pkl_file.exists():
                    self.data = self.load(self.T, self.logg, self.z)
                else:
                    self.data = pd.read_csv(
                        "model/spec/{0}".format(self.filename),
                        delim_whitespace=True,
                        names=["wave", "flux", "nflux"],
                    )
                    del self.data["nflux"]
                    self.save()
            if norm:
                self.data = pd.read_csv(
                    "model/spec/{0}".format(self.filename),
                    delim_whitespace=True,
                    names=["wave", "flux", "nflux"],
                )
                self.data.flux = self.data.nflux
                del self.data["nflux"]
        else:
            self.data = data

    def save(self):
        p = Path("model/pickle/")
        if not p.is_dir():
            p.mkdir()
        pickle.dump(
            self.data,
            open(
                "model/pickle/T{0}g{1:.1f}z{2:0.1f}.pkl".format(
                    self.T, self.logg, self.z
                ),
                "wb",
            ),
        )

    @staticmethod
    def load(T, logg, z):
        data = pickle.load(
            open("model/pickle/T{0}g{1:.1f}z{2:0.1f}.pkl".format(T, logg, z), "rb")
        )
        return data

    def locate_wave(self, wave):
        index = np.abs(self.data.wave - wave).idxmin()
        return index

    def slice_wave(self, wave_range, edge=20):
        self.data = self.data[
            (self.data.wave > wave_range[0] - edge)
            & (self.data.wave < wave_range[1] + edge)
        ]

    def slice_wave_range(self, wave_range_list, edge=20):
        data_list = list()
        for wave_range in wave_range_list:
            data_list.append(
                self.data[
                    (self.data.wave > wave_range[0] - edge)
                    & (self.data.wave < wave_range[1] + edge)
                ]
            )
        self.data = pd.concat(data_list)

    def quick_rebin(self, bin_step):
        self.data = self.data.iloc[::bin_step]

    def fit_spec_range(self, spec, wave_range_list):
        for wave_range in wave_range_list:
            index_min = spec.locate_wave(wave_range[0])
            index_max = spec.locate_wave(wave_range[1])
            item = self.data.loc[index_min : index_max + 1]
            item_spec = spec.data.loc[index_min : index_max + 1]
            popt = np.polyfit(item.wave.values, item.flux / item_spec.flux, 1)
            pd.set_option("mode.chained_assignment", None)
            self.data.flux.loc[index_min : index_max + 1] /= np.polyval(
                popt, item.wave.values
            )

    def cal_chi_square_log_pos(self, spec):
        chi_square_log_pos = None
        if (spec.data.wave == self.data.wave).all():
            residue = spec.data.flux.values - self.data.flux.values
            chi_square = np.sum(residue ** 2 * spec.data.ivar.values)
            chi_square_log_pos = -1 / 2 * chi_square
        return chi_square_log_pos

    def add_rotational(self, vsini, epsilon=0.6):
        wvl = self.data.wave.values
        flux = self.data.flux.values
        self.data.flux = pyasl.rotBroad(wvl, flux, epsilon, vsini)

    def add_rotational_range(self, vsini, wave_range_list, epsilon=0.6, edge=20):
        for wave_range in wave_range_list:
            index_min = self.locate_wave(wave_range[0] - edge)
            index_max = self.locate_wave(wave_range[1] + edge)
            wvl = self.data.loc[index_min:index_max].wave.values
            flux = self.data.loc[index_min:index_max].flux.values
            self.data.loc[index_min:index_max, "flux"] = pyasl.fastRotBroad(
                wvl, flux, epsilon, vsini
            )

    def add_radial(self, rv):
        redshift = rv / (constants.c.cgs.value / 1e5)
        self.data.wave = self.data.wave.values * (1 + redshift)

    def add_radial_range(self, rv_list, wave_range_list, edge=20):
        for i, wave_range in enumerate(wave_range_list):
            rv = rv_list[i]
            index_min = self.locate_wave(wave_range[0] - edge)
            index_max = self.locate_wave(wave_range[1] + edge)
            item = self.data.loc[index_min : index_max + 1]
            redshift = rv / (constants.c.cgs.value / 1e5)
            self.data.loc[index_min : index_max + 1, "wave"] = item.wave * (
                1 + redshift
            )

    def add_instrumental_range(self, resolution, wave_range_list, edge=20):
        """Add instrumental broadening
        
        Args:
            resolution (float): resolution
            wave_range_list (list): wave ranges
            edge (float): include the boundary edge to calculate
                          (20)
        """
        interval = self.data.wave.iloc[1] - self.data.wave.iloc[0]
        for wave_range in wave_range_list:
            index_min = self.locate_wave(wave_range[0] - edge)
            index_max = self.locate_wave(wave_range[1] + edge)
            sigma = np.mean(wave_range) / resolution / 2.355
            kernel = Gaussian1DKernel(stddev=sigma / interval)
            flux_bin_new = convolve(
                self.data.loc[index_min : index_max + 1].flux.values, kernel
            )
            extra = int(
                (len(flux_bin_new) - len(self.data.loc[index_min : index_max + 1])) / 2
            )
            self.data.at[index_min : index_max + 1, "flux"] = flux_bin_new[extra:-extra]

    def add_instrumental(self, resolution):
        self.data.flux = pyasl.instrBroadGaussFast(
            self.data.wave.values, self.data.flux.values, resolution
        )

    def divide_ratio_range(self, ratio_list, wave_range_list, edge=20):
        for index, wave_range in enumerate(wave_range_list):
            index_left = self.locate_wave(wave_range[0] - edge)
            index_right = self.locate_wave(wave_range[1] + edge)
            item = self.data.loc[index_left : index_right + 1]
            length = len(item)
            popt = np.polyfit(
                [
                    item.wave.iloc[0],
                    item.wave.iloc[int(length / 2)],
                    item.wave.iloc[-1],
                ],
                ratio_list[index],
                2,
            )
            self.data.at[
                index_left : index_right + 1, "flux"
            ] = item.flux.values / np.polyval(popt, item.wave.values)

    def bin_spectrum(self, step=0.05):
        pd.options.mode.chained_assignment = None
        wave_old = self.data.wave.values
        wave_new = np.arange(min(wave_old), max(wave_old), step)
        spec = spectrum.ArraySourceSpectrum(wave=wave_old, flux=self.data.flux.values)
        f = np.ones_like(wave_old)
        filt = spectrum.ArraySpectralElement(wave_old, f, waveunits="angstrom")
        flux_new = observation.Observation(
            spec, filt, binset=wave_new, force=None
        ).binflux
        self.data = pd.DataFrame({"wave": wave_new, "flux": flux_new})

    def vac2air(self):
        self.data.wave = pyasl.vactoair2(self.data.wave)

    def rebin_spec_range(self, spec, wave_range_list):
        wave_old = self.data.wave.values
        spec_old = spectrum.ArraySourceSpectrum(
            wave=wave_old, flux=self.data.flux.values
        )
        wave_new_list = list()
        flux_new_list = list()
        for wave_range in wave_range_list:
            index_min = spec.locate_wave(wave_range[0])
            index_max = spec.locate_wave(wave_range[1])
            wave_new = spec.data.loc[index_min : index_max + 1].wave
            f = np.ones_like(wave_old)
            filt = spectrum.ArraySpectralElement(wave_old, f, waveunits="angstrom")
            flux_new = observation.Observation(
                spec_old, filt, binset=wave_new, force=None
            ).binflux
            wave_new_list.append(wave_new)
            flux_new_list.append(flux_new)
        wave_new = np.concatenate(wave_new_list)
        flux_new = np.concatenate(flux_new_list)
        self.data = pd.DataFrame({"wave": wave_new, "flux": flux_new})
        self.data.set_index(spec.data.index, inplace=True)

    def flux_arm_correct(self):
        df = pd.read_csv("./material/cont.csv", index_col=0)
        self.data.flux = self.data.flux.values * np.interp(
            self.data.wave.values, df.wave, df.cont
        )


class ModelS(Model):
    def __init__(self, teff, logg, feh, data):
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.data = data


if __name__ == "__main__":
    pass
