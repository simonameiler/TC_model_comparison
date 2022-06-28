"""
Adapted for code repository on 2022-06-28

description: Load output from Hazard_CHAZ_a.py and concatenate the files into 
            the final regional hazard sets.

@author: simonameiler
"""
import copy

# import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.util.constants import SYSTEM_DIR

############################################################################

haz_dir = SYSTEM_DIR/"hazard"
cent_str = SYSTEM_DIR.joinpath("centroids_0300as_global.hdf5")

# boundaries of (sub-)basins (lonmin, lonmax, latmin, latmax)
BASIN_BOUNDS = {
    # North Atlantic/Eastern Pacific Basin
    'AP': [-180.0, 0.0, 0.0, 65.0],

    # Indian Ocean Basin
    'IO': [30.0, 100.0, 0.0, 40.0],

    # Southern Hemisphere Basin
    'SH': [-180.0, 180.0, -60.0, 0.0],

    # Western Pacific Basin
    'WP': [100.0, 180.0, 0.0, 65.0],
}

reg_id = {'AP': 5000, 'IO': 5001, 'SH': 5002, 'WP': 5003}

# Initiate CHAZ tracks
def basin_split_haz(hazard, basin):
    """ Split CHAZ global hazard up into ocean basins of choice """
    tc_haz_split = TropCyclone()
    # get basin bounds
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[str(basin)]
    basin_idx = (hazard.centroids.lat > y_min) & (
                 hazard.centroids.lat < y_max) & (
                 hazard.centroids.lon > x_min) & (
                 hazard.centroids.lon < x_max)
    hazard.centroids.region_id[basin_idx] = reg_id[basin]
    tc_haz_split = hazard.select(reg_id=reg_id[basin]) 
    return tc_haz_split

# load centroids from this source
cent = Centroids.from_hdf5(cent_str)

# load all CHAZ hazard files and append to list
CHAZ_hazard = []
for i_file in range(1):
    for i_ens in range(3):
        tc_hazard = TropCyclone.from_hdf5(haz_dir.joinpath(
            f"TC_{i_file}_{i_ens}_0300as_CHAZ_ERA5.hdf5"))
        CHAZ_hazard.append(tc_hazard)

# create master TropCyclone object of hazard list; save
CHAZ_master = copy.deepcopy(CHAZ_hazard[0])
for haz in range(1,len(CHAZ_hazard)):
    CHAZ_master.append(CHAZ_hazard[haz])    
CHAZ_master.write_hdf5(haz_dir.joinpath('TC_global_0300as_CHAZ_ERA5.hdf5'))

# call basin split function and save results
for bsn in BASIN_BOUNDS:
    CHAZ_basin = TropCyclone()
    CHAZ_basin = basin_split_haz(CHAZ_master, bsn)
    CHAZ_basin.write_hdf5(haz_dir.joinpath(f"TC_{bsn}_0300as_CHAZ_ERA5.hdf5"))
