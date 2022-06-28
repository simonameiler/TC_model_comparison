"""
Created on 2022-01-13

description: hazard analysis - extract max wind speeds over land (!) from wind field data sets

@author: Thomas Vogt (tvogts)
"""

import pathlib
import sys

import numpy as np

from climada.hazard import TropCyclone

DATA_DIR = pathlib.Path("./data")
OUTPUT_DIR = DATA_DIR / "max_winds"

SYSTEM_DIR = pathlib.Path("/cluster/work/climate/meilers/climada/data")
HAZARD_DIR = SYSTEM_DIR / "hazard"

DATASETS = ['IBTrACS', 'IBTrACS_old', 'IBTrACS_p', 'STORM', 'MIT', 'CHAZ_ERA5']

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

def main(reg):
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[str(reg)]
    for tr in DATASETS:
        out_path = OUTPUT_DIR / f"TC_{reg}_{tr}.npz"
        if out_path.exists():
            print(f"File exists: {out_path}, skipping...")
            continue

        haz_path = HAZARD_DIR / f"TC_{reg}_0300as_{tr}.hdf5"
        haz = TropCyclone.from_hdf5(haz_path)

        haz.centroids.region_id = (
            (haz.centroids.lat > y_min)
            & (haz.centroids.lat < y_max)
            & (haz.centroids.lon > x_min)
            & (haz.centroids.lon < x_max)
            & haz.centroids.on_land.astype(bool)
        ).astype(int)
        haz = haz.select(reg_id=1)

        print(f"Writing to {out_path} ...")
        np.savez_compressed(out_path, intensity=haz.intensity.max(axis=1).toarray().ravel())

if __name__ == "__main__":
    for reg in BASIN_BOUNDS.keys():
        main(reg)
