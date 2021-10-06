from glob import glob

import numpy as np
import pandas as pd

from scipy.interpolate import griddata

from .util import z2mh


class Geneva:
    """Geneva rotation model

    Limit to 1.7 solar mass
    """

    def __init__(self):
        self.Z = np.array([0.002, 0.006, 0.014])
        self.mh_array = z2mh(self.Z)
        self.omega = np.array([0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.data_dir = "./material/Geneva/"

    def retrieve_file(self, Z, omega, mass_str=""):
        Z_q = self.Z[np.abs(self.Z - Z).argmin()]
        omega_q = self.omega[np.abs(self.omega - omega).argmin()]
        omega_q = str(int(omega_q * 10))
        if omega_q == "10":
            omega_q = "c"
        file_names = glob(
            self.data_dir
            + "tablesBeZ{0:0>3d}".format(int(Z_q * 1000))
            + "/M{0}*V{1}.dat".format(mass_str, omega_q)
        )
        return file_names

    def read_Z_omega(self, Z, omega, mass_str=""):
        file_names = self.retrieve_file(Z, omega, mass_str)
        df_list = list()
        for file_name in file_names:
            df = self.read_file(file_name)
            df = df[df["1H_surf"] > 0.69]
            df = df.assign(
                age_rel=(df.time - np.min(df.time))
                / (np.max(df.time) - np.min(df.time))
            )
            df_list.append(df)
        df = pd.concat(df_list)
        return df

    @staticmethod
    def read_file(file_name):
        return pd.read_table(file_name, delim_whitespace=True, skiprows=[1])

    def estimate_vcrit(self, Z, omega, mass, age_rel):
        """Estimate v_crit"""
        df = self.read_Z_omega(Z, omega)
        v_crit = griddata(
            (df.mass, df.age_rel), df.v_crit1, (mass, age_rel), method="linear"
        )
        return v_crit

    def get_mh_index(self, fehs):
        mh_index = np.zeros_like(fehs)
        for i, feh in enumerate(fehs):
            mh_index[i] = np.abs(self.mh_array - feh).argmin()
        return mh_index


if __name__ == "__main__":
    pass
