"""
Adapted for code repository on 2022-06-28

description: Load TC tracks from the STORM model and calculate the 2D windfield
            after Holland (2008).

@author: simonameiler
"""

import os
import numpy as np
from pathlib import Path

# import CLIMADA modules:
from climada.util.constants import SYSTEM_DIR
from climada.hazard import Centroids, TCTracks, TropCyclone

STORM_DIR = SYSTEM_DIR/"tracks"/"STORM"
HAZARD_DIR = SYSTEM_DIR/"hazard"
CENT_STR = SYSTEM_DIR.joinpath("centroids_0300as_global.hdf5")

reg = 'SH' # other regions: 'AP', 'IO', 'SH'
# BASIN_S are as follows:
# 'AP' = ['NA','EP']; 'IO' = 'NI' , WP = 'WP'
BASIN_S = ['SI','SP'] # str or list of basins for STORM tracks
storm_str = f"TC_tracks_STORM_{reg}.p"

def init_STORM_tracks(basin):
    """ Load all STORM tracks for the basin of interest."""
    all_tracks = []
    for j in basin:
        fname = lambda i: f"STORM_DATA_IBTRACS_{j}_1000_YEARS_{i}.txt"
        for i in range(10):
            tracks_STORM = TCTracks.from_simulations_storm(os.path.join(STORM_DIR, fname(i)))
            all_tracks.extend(tracks_STORM.data)
    tracks_STORM.data = all_tracks
    tracks_STORM.equal_timestep(time_step_h=1.)
    return tracks_STORM

def init_tc_hazard(tracks, cent, load_haz=False):
    """initiate TC hazard from tracks and exposure"""
     # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_{reg}_0300as_STORM.hdf5"
    if load_haz and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_hazard = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
    else:
        print("----------------------Initiating Hazard----------------------")
        # hazard is initiated from tracks, windfield computed:
        tc_hazard = TropCyclone.from_tracks(tracks, centroids=cent)
        freq_corr_STORM = 1/10000
        tc_hazard.frequency = np.ones(tc_hazard.event_id.size) * freq_corr_STORM
        tc_hazard.check()
        tc_hazard.write_hdf5(HAZARD_DIR.joinpath(haz_str))
    return tc_hazard

############################ Call functions ##################################
cent = Centroids.from_hdf5(CENT_STR)

tracks_STORM = init_STORM_tracks(basin=BASIN_S)

hazard = init_tc_hazard(tracks_STORM, cent, load_haz=False)
