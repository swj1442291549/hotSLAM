import re
import glob
import pickle

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from scipy.interpolate import griddata

from .util import z2mh


class Evo:
    def __init__(self):
        self.info = None
        self.read_evo_info()
        self.cal_Z_array()
        self.info = self.info.assign(mh=z2mh(self.info.Z))

    def cal_Z_array(self):
        self.Z_array = np.sort(np.array(list(Counter(self.info.Z).keys())))
        self.mh_array = z2mh(self.Z_array)

    @staticmethod
    def read_evo_file(file_name, is_ms=True):
        df = pd.read_csv(file_name, delim_whitespace=True)
        if is_ms:
            df = df[(df.PHASE >= 5) & (df.PHASE <= 6)]
        return df

    def read_evo_info(self):
        mass_list = list()
        age_rel_list = list()
        age_list = list()
        logT_list = list()
        logL_list = list()
        Z_list = list()
        for file_name in sorted(glob.glob("material/cmd_evo/*.DAT")):
            split = re.split("Z|Y|M|.DAT", Path(file_name).stem)
            if 1.5 <= float(split[3]) < 6.4 and float(split[3]) != 1.925:
                evo_data = self.read_evo_file(file_name)
                age_rel_list.append(
                    (evo_data.AGE.values - np.min(evo_data.AGE))
                    / (np.max(evo_data.AGE) - np.min(evo_data.AGE))
                )
                age_list.append(evo_data.AGE.values)
                mass_list.append(evo_data.MASS)
                logL_list.append(evo_data.LOG_L)
                logT_list.append(evo_data.LOG_TE)
                Z_list.append(float(split[1]) * np.ones_like(evo_data.LOG_L))
        mass = np.concatenate(mass_list)
        Z = np.concatenate(Z_list)
        age_rel = np.concatenate(age_rel_list)
        age = np.concatenate(age_list)
        logL = np.concatenate(logL_list)
        logT = np.concatenate(logT_list)
        self.info = pd.DataFrame(
            {
                "mass": mass,
                "age_rel": age_rel,
                "logL": logL,
                "logT": logT,
                "Z": Z,
                "age": age,
            }
        )

    def estimate_mass_age(self, logT, logL, feh):
        self.info = self.info.iloc[::5]
        points = np.vstack(
            (self.info.logT.values, self.info.logL.values, self.info.mh.values)
        ).T
        print(" --- mass ---")
        mass = griddata(points, self.info.mass.values, (logT, logL, feh))
        print(" --- age_rel ---")
        age_rel = griddata(points, self.info.age_rel.values, (logT, logL, feh))
        print(" --- age ---")
        age = griddata(points, self.info.age.values, (logT, logL, feh))
        return mass, age_rel, age

    def get_evo_Z(self, Z):
        return self.Z_array[np.abs(self.Z_array - Z).argmin()]

    def gen_boundary_Z(self):
        boundary_dict = {}
        for Z in self.Z_array:
            info = self.info[self.info.Z == Z]
            points_upper = list()
            info_sel = info[info.mass == max(info.mass)]
            for i in range(len(info_sel)):
                item = info_sel.iloc[i]
                points_upper.append([item.logT, item.logL])
            points_evo_upper = list()
            info_sel = info[info.age_rel == 1]
            for i in range(len(info_sel)):
                item = info_sel.iloc[-i]
                points_evo_upper.append([item.logT, item.logL])
            points_lower = list()
            info_sel = info[info.mass == min(info.mass)]
            for i in range(len(info_sel)):
                item = info_sel.iloc[-i]
                points_lower.append([item.logT, item.logL])
            points_evo_lower = list()
            info_sel = info[info.age_rel == 0]
            for i in range(len(info_sel)):
                item = info_sel.iloc[i]
                points_evo_lower.append([item.logT, item.logL])
            points = np.concatenate(
                [points_lower[1:], points_evo_lower, points_upper, points_evo_upper[1:]]
            )
            boundary_dict[Z] = points
        return boundary_dict

    def save_boundary(self):
        boundary_dict = self.gen_boundary_Z()
        pickle.dump(boundary_dict, open("./material/boundary.pkl", "wb"))

    @staticmethod
    def read_boundary_Z(Z):
        boundary_dict = pickle.load(open("./material/boundary.pkl", "rb"))
        Z_array = np.array(list(boundary_dict.keys()))
        boundary = boundary_dict[Z_array[np.abs(Z_array - Z).argmin()]]
        return boundary

    @staticmethod
    def read_boundary_mh(feh):
        boundary_dict = pickle.load(open("./material/boundary.pkl", "rb"))
        Z_array = np.array(list(boundary_dict.keys()))
        mh_array = z2mh(Z_array)
        boundary = boundary_dict[Z_array[np.abs(mh_array - feh).argmin()]]
        return boundary

    def get_mh_index(self, fehs):
        mh_index = np.zeros_like(fehs)
        for i, feh in enumerate(fehs):
            mh_index[i] = np.abs(self.mh_array - feh).argmin()
        return mh_index


if __name__ == "__main__":
    pass
