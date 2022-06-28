"""
Adapted for code repository on 2022-06-28

description: Impact calculation - basic setup

@author: simonameiler
"""

# load modules
import sys
import numpy as np

# import CLIMADA modules:
from climada.util.constants import SYSTEM_DIR 
from climada.hazard import Centroids, TropCyclone
from climada.entity.exposures.litpop import LitPop
from climada.engine import Impact
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone

def main(reg):

    impact_dir = SYSTEM_DIR/"impact"
    hazard_dir = SYSTEM_DIR/"hazard"
    cent_str = SYSTEM_DIR.joinpath("centroids_0300as_global.hdf5")
    
    # load exposure
    exp = LitPop()
    ent_str = f"litpop_0300as_2014_{reg}.hdf5"
    exp.read_hdf5(SYSTEM_DIR.joinpath(ent_str))
    
    # load centroids
    cent = Centroids()
    cent.read_hdf5(cent_str)
    
    # load hazard sets and create hazard dictionary
    hazard_dict = dict()
    for tr in ['IBTrACS','IBTrACS_p','MIT','STORM','CHAZ_ERA5']:
        tc_haz = TropCyclone()
        haz_str = f"TC_{reg}_0300as_{tr}.hdf5"
        tc_haz.read_hdf5(hazard_dir.joinpath(haz_str))
        print(tc_haz.frequency.sum())
        hazard_dict[tr] = tc_haz
    
    def freq_bias_corr(hazard, years):
        num_tracks = hazard.intensity.max(axis=1).getnnz(axis=0)[0]
        freq_IB = hazard_dict['IBTrACS'].frequency.sum()
        cor = freq_IB/(num_tracks/years)
        freq_corr = cor/years
        hazard.frequency = np.ones(
            hazard.event_id.size)*freq_corr
        return hazard.frequency, cor
    
    # frequency correction where needed
    # CHAZ
    hazard_dict['CHAZ_ERA5'].frequency, cor_c = freq_bias_corr(hazard_dict['CHAZ_ERA5'], 15600)

    #MIT
    freq_corr = {'AP': 9.4143, 'IO': 3.0734, 'SH': 4.4856, 'WP': 10.6551}
    hazard_dict['MIT'].frequency = np.ones(
        hazard_dict['MIT'].event_id.size)*freq_corr[reg]/hazard_dict['MIT'].size
    
    # STORM
    freq_corr_STORM = 1/10000
    hazard_dict['STORM'].frequency = np.ones(
        hazard_dict['STORM'].event_id.size)*freq_corr_STORM
    
    # prepare impact calcuation - after Samuel Eberenz
    # The iso3n codes need to be consistent with the column “region_id” in the 
    # 1. Init impact functions:
    impact_func_set = ImpfSetTropCyclone()
    impact_func_set.set_calibrated_regional_ImpfSet(calibration_approach='RMSF') 
    # get mapping: country ISO3n per region:
    iso3n_per_region = impf_id_per_region = ImpfSetTropCyclone.get_countries_per_region()[2]
    
    code_regions = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5, 'WP1': 6, \
                    'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}
    
    # match exposure with correspoding impact function
    for calibration_region in impf_id_per_region:
        for country_iso3n in iso3n_per_region[calibration_region]:
            exp.gdf.loc[exp.gdf.region_id== country_iso3n, 'if_TC'] = code_regions[calibration_region]
            exp.gdf.loc[exp.gdf.region_id== country_iso3n, 'if_'] = code_regions[calibration_region]
    
    # calculate impact, save as csv files and write dictionary
    impact_dict = dict()
    for imp in hazard_dict:
        impact = Impact()
        imp_str = f"TC_{reg}_impact_{imp}.csv"
        exp.assign_centroids(hazard_dict[imp])
        impact.calc(exp, impact_func_set, hazard_dict[imp], save_mat=True)
        impact.write_csv(impact_dir.joinpath(imp_str))
        impact_dict[imp] = impact

if __name__ == "__main__":
    main(*sys.argv[1:])



        
        
