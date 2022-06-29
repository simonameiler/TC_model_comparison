"""
Created on 2022-01-13

description: hazard analysis - landfall intensity statistics from windfield data

@author: Thomas Vogt (thomas.vogt@pik-potsdam.de)
"""

import pathlib
import pickle
import tarfile
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climada.hazard.tc_tracks import SAFFIR_SIM_CAT
from climada.hazard.trop_cyclone import KN_TO_MS
from climada.util.constants import SYSTEM_DIR

DATASETS = ['IBTrACS', 'IBTrACS_old', 'IBTrACS_p', 'STORM', 'MIT', 'CHAZ_ERA5']
REGIONS = ['AP', 'IO', 'SH', 'WP']

DATA_DIR = pathlib.Path("./data")
SUBSAMPLE_DIR = DATA_DIR / "subsamples"
MAXWIND_DIR = DATA_DIR / "max_winds"
STATS_DIR = DATA_DIR / "stats"

HAZARD_DIR = SYSTEM_DIR / "hazard"

SAFFIR_SIM_CAT_MS = [v * KN_TO_MS for v in SAFFIR_SIM_CAT]
# legacy, used by Simona's scripts:
# SAFFIR_SIM_CAT_MS = [18, 33, 43, 50, 59, 70, 120]


def read_subsamples(path):
    with open(path, "r") as fp:
        subsamples = [l.strip().split(",") for l in fp.readlines()]
    return subsamples


def read_max_wind(name, region):
    path = HAZARD_DIR / f"TC_{region}_0300as_{name}.hdf5"
    h5f = h5py.File(path, 'r')
    event_names = h5f['event_name'][()]
    max_winds = np.load(MAXWIND_DIR / f"TC_{region}_{name}.npz")['intensity']
    return np.array(event_names), max_winds


def take_subsample(names_sorted, names_sorted_idx, subsample):
    subsample = np.array(subsample)
    match_idx = np.searchsorted(names_sorted, subsample)
    mask = (match_idx >= 0) & (match_idx < names_sorted.size)
    if not np.all(mask):
        raise ValueError("Subsample out of bounds:", ", ".join(subsample[~mask].tolist()))
    mask = (names_sorted[match_idx] == subsample)
    if not np.all(mask):
        raise ValueError("In subsample, but not in names:", ", ".join(subsample[~mask].tolist()))
    return names_sorted_idx[match_idx]


def filter_subsamples(hazard, subsamples):
    """Restrict to subsamples with positive max_wind in the region according to the wind field"""
    event_names, max_winds = hazard
    event_names_sorted_idx = np.argsort(event_names)
    event_names_sorted = event_names[event_names_sorted_idx]
    subsamples_idx = [take_subsample(event_names_sorted, event_names_sorted_idx, sample)
                      for sample in subsamples]
    return [
        [name for i, name in zip(idx, names) if max_winds[i] > 0]
        for names, idx in zip(subsamples, subsamples_idx)
    ]


def filter_events(hazard, subsamples_idx):
    """Restrict to events that are contained in at least one subsample"""
    filtered_idx = np.unique(np.concatenate(subsamples_idx))
    haz_filtered = (hazard[0][filtered_idx], hazard[1][filtered_idx])
    idx_corr = np.full(hazard[0].size, -1)
    idx_corr[filtered_idx] = np.arange(filtered_idx.size)
    subsamples_idx_corr = [idx_corr[sample] for sample in subsamples_idx]
    assert all(all(s != -1 for s in sample) for sample in subsamples_idx_corr)
    return haz_filtered, subsamples_idx_corr


def derive_stats(max_winds, subsamples_idx):
    x0_multi = [max_winds[sample] for sample in subsamples_idx]
    intensity_means = np.array([arr.mean() for arr in x0_multi])
    freq, _, _ = plt.hist(x0_multi, bins=SAFFIR_SIM_CAT_MS, density=True)

    # plt.hist automatically squeezes length-1 dimensions:
    if freq.ndim == 1:
        freq = freq[None, :]

    mean_freq = np.mean(freq, axis=0)
    freq_normed = freq / np.fmax(1e-10, mean_freq[None])
    std_freq = np.std(freq, axis=0)
    return intensity_means, mean_freq, freq_normed, std_freq


def get_haz_subsamples(hazard, region, name):
    subsamples = read_subsamples(SUBSAMPLE_DIR / f"TC_{region}_{name}.txt")
    subsamples = filter_subsamples(hazard, subsamples)
    event_names = hazard[0]
    event_names_sorted_idx = np.argsort(event_names)
    event_names_sorted = event_names[event_names_sorted_idx]
    subsamples_idx = [take_subsample(event_names_sorted, event_names_sorted_idx, sample)
                      for sample in subsamples]
    return filter_events(hazard, subsamples_idx)


def main(tr, reg):
    out_path = STATS_DIR / f"haz_hist_{reg}_{tr}.npz"
    if out_path.exists():
        print(f"File exists: {out_path}, skipping...")
        return

    hazard, subsamples_idx = get_haz_subsamples(read_max_wind(tr, reg), reg, tr)
    ints, fq, freq_normed, std = derive_stats(hazard[1], subsamples_idx)

    print(f"Writing to {out_path}...")
    print(tr, reg, ints.mean(), ints.std())
    np.savez_compressed(out_path, intensity_means=ints, fq=fq, freq_normed=freq_normed, std=std)


if __name__ == "__main__":
    for tr in DATASETS:
        for reg in REGIONS:
            main(tr, reg)
