from pathlib import Path

import numpy as np
import pandas as pd

# from berliner.parsec import CMD
from scipy.interpolate import griddata


class Isoc:
    def __init__(self, file_name):
        p = Path(file_name)
        if p.is_file():
            self.load_table(file_name)
        else:
            self.download_cmd(file_name)

    def load_table(self, file_name):
        self.info = pd.read_csv(file_name, index_col=0)

    def save_csv(self, file_name):
        self.info.to_csv(file_name)

    def download_cmd(self, file_name):
        c = CMD()
        grid_logage = (6, 9.5, 0.1)
        grid_mh = (-2, 0.9, 0.1)
        isoc_lgage, isoc_mhini, isoc_list = c.get_isochrone_grid_mh(
            grid_logage=grid_logage,
            grid_mh=grid_mh,
            photsys_file="gaia_tycho2_2mass",
            n_jobs=50,
            verbose=10,
        )

        df_list = list()
        for isoc in isoc_list:
            df = isoc.to_pandas()
            df_list.append(
                df[(df.logg > 3) & (df.logTe > 3.5) & (df.logTe < 4.4) & (df.label > 0)]
            )
        self.info = pd.concat(df_list, ignore_index=True)
        self.save_csv(file_name)

    def estimate_gaia_color_intrinsic(self, logT, logg, feh):
        points = np.vstack(
            (self.info.logTe.values, self.info.logg.values, self.info.MH.values)
        ).T
        color = griddata(
            points,
            self.info.G_BPmag.values - self.info.G_RPmag.values,
            (logT, logg, feh),
        )
        return color


if __name__ == "__main__":
    pass
