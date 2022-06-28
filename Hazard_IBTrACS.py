"""
Adapted for code repository on 2022-06-28

description: Load TC tracks from the IBTrACS and generate a probabilistic track
            set from it. Next, calculate the 2D windfields after Holland (2008)
            for both track sets.

@author: simonameiler
"""

import pickle
import copy as cp

# import CLIMADA modules:
from climada.util.constants import SYSTEM_DIR
from climada.util.save import save
from climada.hazard import Centroids, TCTracks, TropCyclone

# paths and directories
IB_tracks_dir = SYSTEM_DIR/"tracks"/"IBTrACS"
IB_synth_dir = SYSTEM_DIR/"tracks"/"IBTrACS_p"
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

# function to split hazard set up per basin
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

######### read IBTrACS from NetCDF and save as folder of .nc files ###########

# populate tracks by loading data from NetCDF:
tracks = TCTracks.from_ibtracs_netcdf(year_range=(1980,2018))
tracks_IB = TCTracks()
for i in range(0,6):
    filterdict = {'category': i}
    tracks_IB.data.extend(tracks.subset(filterdict).data)
# post processing, increase time steps for smoother wind field:
tracks_IB.equal_timestep(time_step_h=1., land_params=False)
tracks_IB.write_netcdf(IB_tracks_dir)


####### generate probabilistic IBTrACS dataset from historical IBTrACS #######

# load global IBTrACS
tracks_IB = TCTracks.from_netcdf(IB_tracks_dir)

# generate probabilistic tracks
IB_tracks_synth = cp.deepcopy(tracks_IB)
IB_tracks_synth.data = [x for x in IB_tracks_synth.data if x.time.size > 1]
IB_tracks_synth.calc_perturbed_trajectories(nb_synth_tracks=99)
IB_tracks_synth.write_netcdf(IB_synth_dir)


# load centroids
cent = Centroids.from_hdf5(cent_str)

# calculate windfield for the historical IBTrACS
tc_haz = TropCyclone.from_tracks(tracks_IB, centroids=cent)
tc_haz.write_hdf5(haz_dir.joinpath("TC_global_0300as_IBTrACS.hdf5"))


# call basin split function and save results
for bsn in BASIN_BOUNDS:
    tc_haz_basin = TropCyclone()
    tc_haz_basin = basin_split_haz(tc_haz, bsn)
    tc_haz_basin.write_hdf5(haz_dir.joinpath(f"TC_{bsn}_0300as_IBTrACS.hdf5"))

# calculate windfield for the probabilistic IBTrACS
tc_haz = TropCyclone.from_tracks(IB_tracks_synth, centroids=cent)
tc_haz.write_hdf5(haz_dir.joinpath("TC_global_0300as_IBTrACS_p.hdf5"))


# call basin split function and save results
for bsn in BASIN_BOUNDS:
    tc_haz_basin = TropCyclone()
    tc_haz_basin = basin_split_haz(tc_haz, bsn)
    tc_haz_basin.write_hdf5(haz_dir.joinpath(f"TC_{bsn}_0300as_IBTrACS_p.hdf5"))
