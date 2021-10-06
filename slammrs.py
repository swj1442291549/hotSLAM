import warnings
import pickle

from pathlib import Path

import click
import joblib
import pandas as pd

from slam import Slam
from tqdm import tqdm
from astropy.io import fits
from laspec.normalization import normalize_spectrum_general

from catalogue import Cat
from util.mrs import (
    check_filepath,
    prepare_mrs_data,
    read_mrs_single_spectrum,
    read_mrs_spectrum,
)
from util.values import *
from util.atlasmodel import Model, ModelS

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def save_model_spectra_parallel(flats, output_file_name):
    nobs = len(flats)
    ms_list = joblib.Parallel(backend="multiprocessing", n_jobs=-1, verbose=10)(
        joblib.delayed(read_model_spectrum)(flats[i][0], flats[i][1], flats[i][2])
        for i in range(nobs)
    )
    joblib.dump(ms_list, output_file_name)


def read_model_spectrum(teff, logg, feh):
    mod = Model(teff, logg, feh)
    mod.slice_wave_range(wave_range_list, edge=30)
    mod.bin_spectrum(0.04)
    mod.slice_wave_range(wave_range_list)
    flux, count = normalize_spectrum_general(
        mod.data.wave.values, mod.data.flux.values, "spline"
    )
    mod.data.flux = flux
    return mod


def save_sync_spectra_parallel(labels, slam_file_name, output_file_name):
    nobs = len(labels)
    s = Slam.load_dump(slam_file_name)
    print("SLAM0 predicting spectra ...")
    fluxes = s.predict_spectra(labels[:, :-1])
    print("Generating sync spectra ...")
    ms_list = joblib.Parallel(backend="multiprocessing", n_jobs=-1, verbose=10)(
        joblib.delayed(gen_sync_spectrum)(
            fluxes[i], labels[i][0], labels[i][1], labels[i][2], labels[i][3]
        )
        for i in tqdm(range(nobs))
    )
    joblib.dump(ms_list, output_file_name)


def gen_sync_spectrum(flux, teff, logg, feh, vsini):
    data = pd.DataFrame({"wave": wave_array, "flux": flux})
    mod = ModelS(teff, logg, feh, data)
    mod.add_rotational_range(vsini, wave_range_list, edge=10)
    mod.add_instrumental_range(7500, wave_range_list, edge=10)
    mod.flux_arm_correct()
    flux, count = normalize_spectrum_general(
        mod.data.wave.values, mod.data.flux.values, "spline"
    )
    mod.data.flux = flux
    return mod


def save_mrs_spectra_parallel(df, folder_path, output_file_name):
    nobs = len(df)
    ms_list = joblib.Parallel(backend="multiprocessing", n_jobs=-1, verbose=10)(
        joblib.delayed(read_mrs_spectrum)(
            Path(folder_path, df.iloc[i]["stem"]),
            df.iloc[i]["rvb"],
            df.iloc[i]["rvr"],
            norm_type="spline",
            niter=3,
            binwidth=20,
        )
        for i in range(nobs)
    )
    joblib.dump(ms_list, output_file_name)


def save_mrs_single_spectra_parallel(df, folder_path, output_file_name):
    nobs = len(df)
    ms_list = joblib.Parallel(backend="multiprocessing", n_jobs=-1, verbose=10)(
        joblib.delayed(read_mrs_single_spectrum)(
            Path(folder_path, df.iloc[i]["stem"]),
            df.iloc[i]["rv_opt"],
            df.iloc[i]["ext_r"],
            df.iloc[i]["ext_b"],
            norm_type="spline",
            niter=3,
        )
        for i in range(nobs)
    )
    joblib.dump(ms_list, output_file_name)


def prepare_model_data(ms_list, wave, snr=None):
    flux_norm = np.array(
        [np.interp(wave, ms.data.wave, ms.data.flux) for ms in ms_list]
    )
    if snr is not None:
        ratio = np.random.normal(0, 1 / snr, flux_norm.reshape(-1).shape).reshape(
            flux_norm.shape[0], flux_norm.shape[1]
        )
        flux_norm *= ratio + 1
    ivar_norm = np.ones_like(flux_norm) * 10000
    return flux_norm, ivar_norm


def train_slam(wave, flux_norm, ivar_norm, labels, output_file_name):
    s = Slam(
        wave,
        tr_flux=flux_norm,
        tr_ivar=ivar_norm,
        tr_labels=labels,
        scale=True,
        robust=False,
        mask_conv=(1, 2),
        flux_bounds=(0.001, 100.0),
        ivar_eps=0,
    )

    pgrid = {
        "C": 10.0 ** np.array([-1, 0, 1]),
        "gamma": 10.0 ** np.array([-1, 0, 1]),
        "epsilon": [0.05],
    }

    s.train_pixels(
        profile=None,
        targets="all",
        temp_dir=None,
        sample_weight_scheme="bool",
        model="svr",
        method="grid",
        param_grid=pgrid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=40,
        verbose=5,
        backend="multiprocessing",
    )
    s.training_nmse(n_jobs=30)
    s.save_dump(output_file_name)


def generate_param(num):
    logteff = np.random.normal(3.6, 0.3, 100000)
    logteff = logteff[(logteff >= np.log10(6000)) & (logteff <= np.log10(15000))]
    teff = 10 ** np.random.choice(logteff, num)
    logg = np.random.random(num) * 1 + 3.5
    feh = np.random.random(num) * 2 - 1
    vsini = np.random.normal(200, 150, 100000)
    vsini = vsini[(vsini >= 10) & (vsini <= 500)]
    vsini = np.random.choice(vsini, num)
    return pd.DataFrame({"teff": teff, "logg": logg, "feh": feh, "vsini": vsini})


def get_file_name(df):
    stem_list = list()
    for i in range(len(df)):
        item = df.iloc[i]
        stem_list.append(
            "med-{0}-{1}_sp{2:0>2d}-{3:0>3d}.fits.gz".format(
                item.lmjd, item.planid.decode().strip(), item.spid, item.fiberid
            )
        )
    df = df.assign(stem=stem_list)
    return df


@click.command()
@click.argument("tag", type=int)
def main(tag):
    if tag == 1:
        teff = np.arange(6000, 15100, 100)
        logg = np.arange(3.5, 5.1, 0.1)
        feh = np.arange(-1, 1.5, 0.5)

        meshs = np.meshgrid(teff, logg, feh)
        flats = np.array([_.flatten() for _ in meshs]).T

        output_file_name = "dump/slam0_train.dump"
        if not Path(output_file_name).is_file():
            save_model_spectra_parallel(flats, output_file_name)
        ms_list = joblib.load(output_file_name)

        flux_norm, ivar_norm = prepare_model_data(ms_list, wave_array)

        output_file_name = "dump/slam0.dump"
        if not Path(output_file_name).is_file():
            train_slam(wave_array, flux_norm, ivar_norm, flats, output_file_name)

    elif tag == 2:
        slam_file_name = "dump/slam0.dump"

        p = Path("material/slam1_train.csv")
        if not p.is_file():
            param = generate_param(10000)
            param.to_csv(p)
        param = pd.read_csv(p, index_col=0)
        labels = param.to_numpy()

        output_file_name = "dump/slam1_train.dump"
        if not Path(output_file_name).is_file():
            save_sync_spectra_parallel(labels, slam_file_name, output_file_name)
        ms_list = joblib.load(output_file_name)

        flux_norm, ivar_norm = prepare_model_data(ms_list, wave_mrs_array, 1000)

        output_file_name = "dump/slam1.dump"
        if not Path(output_file_name).is_file():
            train_slam(wave_mrs_array, flux_norm, ivar_norm, labels, output_file_name)

    elif tag == 3:
        c = Cat("./data/catalog/dr7_med_clean_sel.fits", "EDR3")
        spec_dir = "./data/mrs"
        check_filepath(c.data, spec_dir)

        output_file_name = "dump/slam1_mrs.dump"
        if not Path(output_file_name).is_file():
            save_mrs_spectra_parallel(c.data, spec_dir, output_file_name)
        ms_list = joblib.load(output_file_name)

        flux_norm, ivar_norm = prepare_mrs_data(ms_list, wave_mrs_array)

        slam_file_name = "dump/slam1.dump"
        s = Slam.load_dump(slam_file_name)

        init_file_name = "result/Xinit_mrs.npy"
        if not Path(init_file_name).is_file():
            Xinit = s.predict_labels_quick(flux_norm, ivar_norm, n_jobs=15)
            np.save(init_file_name, Xinit)
        Xinit = np.load(init_file_name)
        output_folder = "pickle/mrs/"
        temp_file_names = output_folder + c.data.obsid.astype(str) + ".pkl"
        p = Path(output_folder)
        if not p.is_dir():
            p.mkdir(parents=True)

        Rpred = s.predict_labels_multi_filename(
            Xinit, flux_norm, ivar_norm, n_jobs=24, file_names=temp_file_names
        )

    elif tag == 4:
        p = Path("material/slam1_train.csv")
        param = pd.read_csv(p, index_col=0)
        labels = param.to_numpy()

        output_file_name = "dump/slam1_train.dump"
        ms_list = joblib.load(output_file_name)

        for snr in [20, 40, 60, 80, 100]:
            flux_norm, ivar_norm = prepare_model_data(ms_list, wave_mrs_array, snr)

            output_file_name = "dump/slam1.dump"
            s = Slam.load_dump(output_file_name)

            init_file_name = "result/Xinit_snr{0:d}.npy".format(snr)
            if not Path(init_file_name).is_file():
                Xinit = s.predict_labels_quick(
                    flux_norm[::50], ivar_norm[::50], n_jobs=5
                )
                np.save(init_file_name, Xinit)
            Xinit = np.load(init_file_name)
            result_file_name = "result/Xpred_snr{0:d}.npy".format(snr)
            result_std_file_name = "result/Xpred_std_snr{0:d}.npy".format(snr)
            if not Path(result_file_name).is_file():
                Rpred = s.predict_labels_multi(
                    Xinit, flux_norm[::50], ivar_norm[::50], n_jobs=2
                )
                Xpred = np.array([_["x"] for _ in Rpred])
                Xpred_std = np.array([_["pstd"] for _ in Rpred])
                np.save(result_file_name, Xpred)
                np.save(result_std_file_name, Xpred_std)
            Xpred = np.load(result_file_name)
            Xpred_std = np.load(result_std_file_name)
            df = pd.DataFrame(
                Xpred, columns=["teff_slam", "logg_slam", "feh_slam", "vsini_slam"]
            )
            df_std = pd.DataFrame(
                Xpred_std,
                columns=[
                    "teff_slam_std",
                    "logg_slam_std",
                    "feh_slam_std",
                    "vsini_slam_std",
                ],
            )
            param_sel = param.iloc[::50]
            param_sel = param_sel.reset_index(drop=True)
            param_sel = pd.merge(param_sel, df, left_index=True, right_index=True)
            param_sel = pd.merge(param_sel, df_std, left_index=True, right_index=True)
            param_sel.to_csv("result/snr{0:d}.csv".format(snr))


if __name__ == "__main__":
    main()
