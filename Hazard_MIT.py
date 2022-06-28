"""
Adapted for code repository on 2022-06-28

description: Load TC tracks from the MIT model and calculate the 2D windfield
            after Holland (2008).

@author: simonameiler
"""

import sys

# import CLIMADA modules:
from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.util.constants import SYSTEM_DIR

############################################################################

def main(reg):

    mit_dir = SYSTEM_DIR.joinpath('tracks','Kerry')
    haz_dir = SYSTEM_DIR/"hazard"
    haz_str = f"TC_{reg}_0300as_MIT.hdf5"
    
    cent_str = SYSTEM_DIR.joinpath("centroids_0300as_global.hdf5")
    
    # Initiate MIT tracks
    def init_MIT_tracks(model_run, reg):
        fname = mit_dir.joinpath(f"temp_{reg}_era5_reanalcal.mat")
        if reg == 'SH':
            tracks_MIT = TCTracks.from_simulations_emanuel(fname, hemisphere='S')
        else:
            tracks_MIT = TCTracks.from_simulations_emanuel(fname, hemisphere='N')
        tracks_MIT.equal_timestep(time_step_h=1)
        return tracks_MIT
    
    # call functions
    tc_tracks = TCTracks()
    tc_tracks = init_MIT_tracks(reg)
    
    # load centroids from this source
    cent = Centroids.from_hdf5(cent_str)

    tc_hazard = TropCyclone.from_tracks(tc_tracks, centroids=cent)
    tc_hazard.write_hdf5(haz_dir.joinpath(haz_str))
    tc_hazard.check()

if __name__ == "__main__":
    main(*sys.argv[1:])