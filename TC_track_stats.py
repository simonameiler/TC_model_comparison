"""
Created on 2022-01-13

description: hazard analysis - landfall intensity statistics from track data

@author: Thomas Vogt (thomas.vogt@pik-potsdam.de)
"""

import itertools
import logging
import os
import pathlib
import pickle
import sys
import tarfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climada.hazard.tc_tracks import TCTracks, SAFFIR_SIM_CAT, _get_landfall_idx
from climada.hazard.trop_cyclone import KN_TO_MS
import climada.util.coordinates as u_coord
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger('track_stats')
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
if LOGGER.hasHandlers():
    for handler in LOGGER.handlers:
        LOGGER.removeHandler(handler)
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
CONSOLE = logging.StreamHandler(stream=sys.stdout)
CONSOLE.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE)

REGIONS = ['AP', 'IO', 'SH', 'WP']

DATA_DIR = pathlib.Path("./data")
SUBSAMPLE_DIR = DATA_DIR / "subsamples"

MAXWIND_DIR = DATA_DIR / "max_winds"
MAXWIND_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = DATA_DIR / "stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HAZARD_DIR = SYSTEM_DIR / "hazard"
TRACKS_DIR = SYSTEM_DIR / "tracks"

SAFFIR_SIM_CAT_MS = [v * KN_TO_MS for v in SAFFIR_SIM_CAT]

IBTRACS_NAME_CHANGES = {
    # old name : new name
    "2012009S12118": "2012009S12121",
    "2014065S10139": "2014066S09138",
    "2017081S13152": "2017082S14152",
    "2018007S13129": "2018007S13130",
    "2018074S09130": "2018073S09129",
    "2018270S11162": "2018269S09157",
    "2018336S14154": "2018334S08156",
    "2018365S13140": "2018362S13147",
}


def read_tracks_IBTrACS(name="IBTrACS"):
    cache = DATA_DIR / "tracks" / f"{name}.pickle"
    if cache.exists():
        with open(cache, "rb") as fp:
            tracks = pickle.load(fp)
    else:
        tracks = TCTracks.from_netcdf(TRACKS_DIR / name)
        with open(cache, "wb") as fp:
            pickle.dump(tracks, fp)
    return tracks


def read_tracks_STORM(pool=None):
    path = TRACKS_DIR / "STORM"
    fnames = list(path.glob("*.txt"))
    tracks = TCTracks()
    if pool is None:
        all_tracks = [TCTracks.from_simulations_storm(f) for f in fnames]
    else:
        all_tracks = pool.map(TCTracks.from_simulations_storm, fnames)
    tracks.data = sum([t.data for t in all_tracks], [])
    return tracks


def read_tracks_MIT(reg="global"):
    if reg == "global":
        tracks = TCTracks()
        for r in REGIONS:
            tracks_r = read_tracks_MIT(r)
            tracks.data += tracks_r.data
        return tracks

    path = TRACKS_DIR / "Kerry" / f"temp_{reg}_era5_reanalcal.mat"
    tracks = TCTracks.from_simulations_emanuel(path)
    for tr in tracks.data:
        tr.basin.values[:] = reg
    return tracks


def read_tracks_CHAZ_ERA5(pool=None):
    path = TRACKS_DIR / "CHAZ" / "ERA-5"
    fnames = list(path.glob("*.nc"))
    if pool is None:
        return TCTracks.from_simulations_chaz(fnames)
    all_tracks = pool.map(TCTracks.from_simulations_chaz, fnames)
    tracks = TCTracks()
    tracks.data = sum([t.data for t in all_tracks], [])
    return tracks


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


def filter_subsamples(name, region, subsamples):
    """Restrict to subsamples with positive max_wind in the region according to the wind field"""
    event_names, max_winds = read_max_wind(name, region)
    event_names_sorted_idx = np.argsort(event_names)
    event_names_sorted = event_names[event_names_sorted_idx]
    subsamples_idx = [take_subsample(event_names_sorted, event_names_sorted_idx, sample)
                      for sample in subsamples]
    return [
        [name for i, name in zip(idx, names) if max_winds[i] > 0]
        for names, idx in zip(subsamples, subsamples_idx)
    ]


def filter_tracks(tracks, subsamples_idx):
    """Restrict to tracks that are contained in at least one subsample"""
    filtered_idx = np.unique(np.concatenate(subsamples_idx))
    tracks_filtered = TCTracks()
    tracks_filtered.data = [tracks.data[i] for i in filtered_idx]
    idx_corr = np.full(tracks.size, -1)
    idx_corr[filtered_idx] = np.arange(filtered_idx.size)
    subsamples_idx_corr = [idx_corr[sample] for sample in subsamples_idx]
    assert all(all(s != -1 for s in sample) for sample in subsamples_idx_corr)
    return tracks_filtered, subsamples_idx_corr


def add_dist_to_coast(tracks):
    lats = np.concatenate([d.lat.values for d in tracks.data])
    lons = np.concatenate([d.lon.values for d in tracks.data])
    dists = u_coord.dist_to_coast_nasa(lats, lons, highres=True, signed=True)
    LOGGER.info("done: read dist to coast")
    splits = np.cumsum([d.lat.values.size for d in tracks.data])[:-1]
    LOGGER.info("done: get splits for dist to coast")
    for tr, dist in zip(tracks.data, np.split(dists, splits)):
        tr['dist_to_coast'] = ('time', dist)
    LOGGER.info("done: write dist to coast")


def add_lf_positions(track, dist_coast_thresh_km):
    track['on_land'] = (track.dist_to_coast <= dist_coast_thresh_km * 1000)
    sea_land_idx, _ = _get_landfall_idx(track, True)

    track['sea_land_mask'] = ('time', np.full_like(track.time.values, False, dtype=bool))
    track.sea_land_mask.values[sea_land_idx] = True

    # landfall : positive values over land, value i for i-th landfall event
    track['landfall'] = ('time', np.full_like(track.time.values, 0, dtype=int))
    track.landfall[track.on_land] = np.cumsum(track.sea_land_mask)[track.on_land]


def get_lf_winds(track, dist_coast_thresh_km):
    add_lf_positions(track, dist_coast_thresh_km)
    return  np.array([
        # max wind attained during each landfall event (whole time spent over land)
        track.max_sustained_wind.values[track.landfall.values == i].max() * KN_TO_MS
        for i in np.arange(1, track.landfall.values.max() + 1)])


def ragged_to_full(ragged):
    """Creates a regular NumPy array with NaNs from a ragged array (i.e. a list of 1-d arrays)"""
    sizes = np.array([r.size for r in ragged])
    out_shape = (len(ragged), sizes.max())
    out = np.full(out_shape, np.nan)
    ii = np.repeat(np.arange(out_shape[0]), sizes)
    jj = np.concatenate([np.arange(s) for s in sizes])
    out[ii, jj] = np.concatenate(ragged)
    return out


def full_to_ragged(array):
    """List of rows as NumPy arrays with NaN-values removed (varying size!)"""
    return [row[~np.isnan(row)] for row in array]


def get_lf_winds_multi(tr, reg, tracks, dist_coast_thresh_km, pool=None):
    path = MAXWIND_DIR / f"TC_{reg}_{tr}_tracks-{dist_coast_thresh_km:d}.npz"
    if path.exists():
        LOGGER.info("get_lf_winds_multi: load")
        return ragged_to_full(np.load(path)['lf_winds'])

    LOGGER.info("get_lf_winds_multi: compute")
    if pool is None:
        max_winds = [get_lf_winds(tr, dist_coast_thresh_km) for tr in tracks.data]
    else:
        chunksize = min(tracks.size // pool.ncpus, 1000)
        max_winds = pool.map(get_lf_winds, tracks.data,
                             itertools.repeat(dist_coast_thresh_km, tracks.size),
                             chunksize=chunksize)

    LOGGER.info("get_lf_winds_multi: store")
    np.savez_compressed(path, lf_winds=ragged_to_full(max_winds))

    return max_winds


def derive_stats(max_winds, subsamples_idx, lf_mode):
    LOGGER.info("derive_stats")
    if lf_mode == "max":
        max_winds = [[arr.max() if arr.size > 0 else 0] for arr in max_winds]

    x0_multi = [np.concatenate([max_winds[i] for i in sample]) for sample in subsamples_idx]
    intensity_means = np.array([arr.mean() for arr in x0_multi])
    freq, _, _ = plt.hist(x0_multi, bins=SAFFIR_SIM_CAT_MS, density=True)

    # plt.hist automatically squeezes length-1 dimensions:
    if freq.ndim == 1:
        freq = freq[None, :]

    mean_freq = np.mean(freq, axis=0)
    freq_normed = freq / np.fmax(1e-10, mean_freq[None])
    std_freq = np.std(freq, axis=0)
    return intensity_means, mean_freq, freq_normed, std_freq


def get_track_subsamples(tracks, region, name):
    subsamples = read_subsamples(SUBSAMPLE_DIR / f"TC_{region}_{name}.txt")
    subsamples = filter_subsamples(name, region, subsamples)

    track_names = np.array([tr.sid for tr in tracks.data])

    if name == "IBTrACS_old":
        # some IBTrACS storms changed their IDs at some point:
        for old_name, new_name in IBTRACS_NAME_CHANGES.items():
            track_names[track_names == new_name] = old_name

    track_names_sorted_idx = np.argsort(track_names)
    track_names_sorted = track_names[track_names_sorted_idx]
    subsamples_idx = [take_subsample(track_names_sorted, track_names_sorted_idx, sample)
                      for sample in subsamples]
    return filter_tracks(tracks, subsamples_idx)


def region_stats(tracks_global, tr, reg, out_path):
    if reg != "global":
        tracks, subsamples_idx = get_track_subsamples(tracks_global, reg, tr)
    else:
        tracks = tracks_global

    stats = {}
    for thresh_km in [0, 50, 100, 150, 200, 250, 300]:
        for mode in ["max", "all"]:
            max_winds = get_lf_winds_multi(tr, reg, tracks, thresh_km, pool=pool)
            stats[f'numtracks_{mode}_{thresh_km:d}'] = len(max_winds)
            stats[f'numlandfalls_{mode}_{thresh_km:d}'] = sum([
                    w.size > 0 if mode == "max" else w.size for w in max_winds])
            max_winds_ts = [w[w >= 17.5] for w in max_winds]
            stats[f'numlandfalls_ts_{mode}_{thresh_km:d}'] = sum([
                    w.size > 0 if mode == "max" else w.size for w in max_winds_ts])

            if reg != "global":
                ints, fq, freq_normed, std = derive_stats(max_winds, subsamples_idx, mode)
                stats[f'intensity_means_{mode}_{thresh_km:d}'] = ints
                stats[f'fq_{mode}_{thresh_km:d}'] = fq
                stats[f'freq_normed_{mode}_{thresh_km:d}'] = freq_normed
                stats[f'std_{mode}_{thresh_km:d}'] = std

    LOGGER.info(f"Writing to {out_path}...")
    np.savez_compressed(out_path, **stats)


def main(tr, pool):
    read_fun = {
        'IBTrACS': read_tracks_IBTrACS,
        'IBTrACS_old': read_tracks_IBTrACS,
        'IBTrACS_p': lambda: read_tracks_IBTrACS(name="IBTrACS_p"),
        'MIT': read_tracks_MIT,
        'STORM': lambda: read_tracks_STORM(pool=pool),
        'CHAZ_ERA5': lambda: read_tracks_CHAZ_ERA5(pool=pool),
    }[tr]

    regions = ["global"] + REGIONS
    reg_paths = [OUTPUT_DIR / f"track_hist_{reg}_{tr}.npz" for reg in regions]

    tracks_global = TCTracks()
    if not all(p.exists() for p in reg_paths):
        cache_path = DATA_DIR / "tracks" / "CHAZ_ERA5.pickle"
        if tr == "CHAZ_ERA5" and cache_path.exists():
            with open(cache_path, "rb") as fp:
                tracks_global = pickle.load(fp)
        else:
            tracks_global = read_fun()
            tracks_global.equal_timestep(time_step_h=1, pool=pool)
            add_dist_to_coast(tracks_global)
            if tr == "CHAZ_ERA5":
                LOGGER.info("Caching")
                with open(cache_path, "wb") as fp:
                    pickle.dump(tracks_global, fp)
                LOGGER.info("Done: Caching")

    for reg, out_path in zip(regions, reg_paths):
        if out_path.exists():
            LOGGER.info(f"File exists: {out_path}, skipping...")
            continue

        tracks = tracks_global
        if tr == "MIT" and reg != "global":
            tracks = TCTracks()
            tracks.data = [track for track in tracks_global.data
                           if track.basin.values[0] == reg]
        region_stats(tracks, tr, reg, out_path)


if __name__ == "__main__":
    pool = None
    if 'OMP_NUM_THREADS' in os.environ:
        from pathos.pools import ProcessPool as Pool
        pool = Pool(nodes=int(os.environ['OMP_NUM_THREADS']))

    main(sys.argv[1], pool)
