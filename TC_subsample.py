"""
Created on 2022-01-12

description: hazard analysis - draw subsamples at length of IBTrACS (39 years)

@author: Thomas Vogt (thomas.vogt@pik-potsdam.de)
"""

import datetime as dt
import pathlib
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from climada.hazard import TropCyclone

REGIONS = ['AP', 'IO', 'SH', 'WP']

DATA_DIR = pathlib.Path("./data")
OUTPUT_DIR = DATA_DIR / "subsamples"

SYSTEM_DIR = pathlib.Path("/cluster/work/climate/meilers/climada/data")
HAZARD_DIR = SYSTEM_DIR / "hazard"

RNG = np.random.default_rng(123456789)


def subsample_IBTrACS(haz):
    return [haz.event_name]


def subsample_IBTrACS_p(haz, step):
    subsamples = []
    for i in np.arange(step):
        event_selected = np.zeros(len(haz.event_name), dtype=bool)
        event_selected[i::step] = True

        subsamples.append([n for i, n in enumerate(haz.event_name) if event_selected[i]])
    return subsamples


def subsample_STORM(haz, N=1000, yrs_hist=39):
    event_year = np.array([
       int(n.split(".")[0][-1]) * 1000 + int(n.split("-")[1])
       for n in haz.event_name])

    subsamples = []
    for draw in range(N):
        draw_years = RNG.choice(
            np.arange(10000), yrs_hist, replace=False)
        event_selected = np.isin(event_year, draw_years)

        subsamples.append([n for i, n in enumerate(haz.event_name) if event_selected[i]])
    return subsamples


def subsample_CHAZ(haz):
    event_ensemble = np.array([
        int(n[17:19]) * 100 + int(n.split("-")[2])
        for n in haz.event_name])

    subsamples = []
    for ensemble in np.unique(event_ensemble):
        event_selected = (event_ensemble == ensemble)

        subsamples.append([n for i, n in enumerate(haz.event_name) if event_selected[i]])
    return subsamples


def subsample_MIT(reg, haz, N=1000):
    # load list of freq_year values from matlab file
    freq_year = loadmat(HAZARD_DIR.joinpath(f"freqyear_{reg}.mat"))['freqyear'][0].tolist()

    subsamples = []
    for draw in range(N):
        event_year = np.array([
            dt.datetime.fromordinal(d).year
            for d in haz.date.astype(int)])

        year_sample_sizes = {
            year: n
            for year, n in zip(range(1979, 2020), RNG.poisson(freq_year))}

        event_selected_idx = np.concatenate([
            RNG.choice((event_year == year).nonzero()[0],
                       size=year_sample_sizes[year],
                       replace=False)
            for year in range(1980, 2019)])

        event_selected = np.zeros(len(haz.event_name), dtype=bool)
        event_selected[event_selected_idx] = True

        subsamples.append([n for i, n in enumerate(haz.event_name) if event_selected[i]])
    return subsamples


def read_haz_meta(name, region):
    path = HAZARD_DIR / f"TC_{region}_0300as_{name}.hdf5"
    h5f = h5py.File(path, 'r')
    haz = TropCyclone()
    haz.event_name = h5f['event_name'][()]
    haz.date = h5f['date'][()]
    return haz


def save_subsamples(subsamples, path):
    assert all(all("," not in s for s in sample) for sample in subsamples)
    print(f"Writing to {path}...")
    with open(path, "w") as fp:
        fp.writelines([",".join(sample) + "\n" for sample in subsamples])


def main(reg):
    subsample_funcs = {
        'IBTrACS': subsample_IBTrACS,
        'IBTrACS_old': subsample_IBTrACS,
        'IBTrACS_p': lambda haz: subsample_IBTrACS_p(haz, 100),
        'STORM': subsample_STORM,
        'MIT': lambda haz: subsample_MIT(reg, haz),
        'CHAZ_ERA5': subsample_CHAZ,
    }

    for tr, subsample_fun in subsample_funcs.items():
        out_path = OUTPUT_DIR / f"TC_{reg}_{tr}.txt"
        if out_path.exists():
            print(f"File exists: {out_path}, skipping...")
            continue

        save_subsamples(
            subsample_fun(read_haz_meta(tr, reg)),
            out_path)

if __name__ == "__main__":
    for reg in REGIONS:
        main(reg)
