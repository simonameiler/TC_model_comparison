"""
Adapted for code repository on 2022-06-28

description: Load TC tracks from the CHAZ model and calculate the 2D windfield
            after Holland (2008).
            Note, job is run on Euler over many files/ensembles. 
            Hazard_CHAZ_b.py is used to concatenate the files into the final 
            regional hazard sets.

@author: simonameiler
"""

import sys

# import CLIMADA modules:
from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.util.constants import SYSTEM_DIR

############################################################################

def main(i_file, i_ens):

    chaz_dir = SYSTEM_DIR.joinpath('tracks','CHAZ', 'ERA-5')
    haz_dir = SYSTEM_DIR/"hazard"
    haz_str = f"TC_{i_file}_{i_ens}_0300as_CHAZ_ERA5.hdf5"
    
    cent_str = SYSTEM_DIR.joinpath("centroids_0300as_global.hdf5")
    
    # Initiate CHAZ tracks
    def init_CHAZ_tracks_ens(i_file, i_ens):
        fname = chaz_dir.joinpath(f"global_2019_2ens00{i_file}_pre.nc")
        tracks_CHAZ = TCTracks.from_simulations_chaz(fname, ensemble_nums=i_ens)
        tracks_CHAZ.equal_timestep(time_step_h=1)
        return tracks_CHAZ
    
    # call functions
    tc_tracks = TCTracks()
    tc_tracks = init_CHAZ_tracks_ens(i_file, i_ens)
    
    # load centroids from this source
    cent = Centroids.from_hdf5(cent_str)

    tc_hazard = TropCyclone.from_tracks(tc_tracks, centroids=cent)
    tc_hazard.write_hdf5(haz_dir.joinpath(haz_str))
    tc_hazard.check()

if __name__ == "__main__":
    main(*sys.argv[1:])