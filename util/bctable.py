import numpy as np

from astropy.table import Table


class BC:
    def __init__(self):
        self.info = None

    def load_fits(self, file_name):
        self.info = Table.read(f"{file_name}").to_pandas()

    def interp_bc(self, logT, logg, av, passband_index):
        """Find corresponding BC for logT, logg and av

        logT (float): log temperature
        logg (float): log surface gravity
        av (float): extinction
        passband_index: index of the desired passband (starting from 0)

        """
        index = round((logT - 3.41) * 100) * 14 + round((logg + 0.5) / 0.5)
        item = self.info.loc[index]
        passband_len = int((self.info.shape[1] - 2) / 7)
        return np.interp(
            av,
            [0, 0.5, 1, 2, 5, 10, 20],
            [item[2 + passband_index + i * passband_len] for i in range(7)],
        )
