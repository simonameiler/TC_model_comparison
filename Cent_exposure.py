"""
Adapted for code repository on 2022-06-28

description: Make exposure files using the LitPop methodology as described in
            Eberenz et al. (2020) and define centroids from exposure.

@author: simonameiler
"""
import numpy as np
from pathlib import Path

# import CLIMADA modules:
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.hazard import Centroids
from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures.base import INDICATOR_CENTR
from climada.util.coordinates import dist_to_coast
from climada.util.coordinates import pts_to_raster_meta, get_resolution

## countries by region:
region_ids_cal = {'NA1': ['AIA', 'ATG', 'ARG', 'ABW', 'BHS', 'BRB', 'BLZ', 
                          'BMU', 'BOL', 'CPV', 'CYM', 'CHL', 'COL', 'CRI', 
                          'CUB', 'DMA', 'DOM', 'ECU', 'SLV', 'FLK', 'GUF', 
                          'GRD', 'GLP', 'GTM', 'GUY', 'HTI', 'HND', 'JAM', 
                          'MTQ', 'MEX', 'MSR', 'NIC', 'PAN', 'PRY', 'PER', 
                          'PRI', 'SHN', 'KNA', 'LCA', 'VCT', 'SXM', 'SUR', 
                          'TTO', 'TCA', 'URY', 'VEN', 'VGB', 'VIR'], \
                  'NA2': ['CAN', 'USA'], \
                  'NI': ['AFG', 'ARM', 'AZE', 'BHR', 'BGD', 'BTN', 'DJI', 
                         'ERI', 'ETH', 'GEO', 'IND', 'IRN', 'IRQ', 'ISR', 
                         'JOR', 'KAZ', 'KWT', 'KGZ', 'LBN', 'MDV', 'MNG', 
                         'MMR', 'NPL', 'OMN', 'PAK', 'QAT', 'SAU', 'SOM', 
                         'LKA', 'SYR', 'TJK', 'TKM', 'UGA', 'ARE', 'UZB', 
                         'YEM'], \
                  'OC': ['ASM', 'AUS', 'COK', 'FJI', 'PYF', 'GUM', 'KIR', 
                         'MHL', 'FSM', 'NRU', 'NCL', 'NZL', 'NIU', 'NFK', 
                         'MNP', 'PLW', 'PNG', 'PCN', 'WSM', 'SLB', 'TLS', 
                         'TKL', 'TON', 'TUV', 'VUT', 'WLF'], \
                  'SI': ['COM', 'COD', 'SWZ', 'MDG', 'MWI', 'MLI', 'MUS', 
                         'MOZ', 'ZAF', 'TZA', 'ZWE'], \
                  'WP1': ['KHM', 'IDN', 'LAO', 'MYS', 'THA', 'VNM'], \
                  'WP2': ['PHL'], \
                  'WP3': ['CHN'], \
                  'WP4': ['HKG', 'JPN', 'KOR', 'MAC', 'TWN'], 
                  'ROW': ['ALB', 'DZA', 'AND', 'AGO', 'ATA', 'AUT', 'BLR', 
                          'BEL', 'BEN', 'BES', 'BIH', 'BWA', 'BVT', 'BRA', 
                          'IOT', 'BRN', 'BGR', 'BFA', 'BDI', 'CMR', 'CAF', 
                          'TCD', 'CXR', 'CCK', 'COG', 'HRV', 'CUW', 'CYP', 
                          'CZE', 'CIV', 'DNK', 'EGY', 'GNQ', 'EST', 'FRO', 
                          'FIN', 'FRA', 'ATF', 'GAB', 'GMB', 'DEU', 'GHA', 
                          'GIB', 'GRC', 'GRL', 'GGY', 'GIN', 'GNB', 'HMD', 
                          'VAT', 'HUN', 'ISL', 'IRL', 'IMN', 'ITA', 'JEY', 
                          'KEN', 'PRK', 'XKX', 'LVA', 'LSO', 'LBR', 'LBY', 
                          'LIE', 'LTU', 'LUX', 'MLT', 'MRT', 'MYT', 'MDA', 
                          'MCO', 'MNE', 'MAR', 'NAM', 'NLD', 'NER', 'NGA', 
                          'MKD', 'NOR', 'PSE', 'POL', 'PRT', 'ROU', 'RUS', 
                          'RWA', 'REU', 'BLM', 'MAF', 'SPM', 'SMR', 'STP', 
                          'SEN', 'SRB', 'SYC', 'SLE', 'SGP', 'SVK', 'SVN', 
                          'SGS', 'SSD', 'ESP', 'SDN', 'SJM', 'SWE', 'CHE', 
                          'TGO', 'TUN', 'TUR', 'UKR', 'GBR', 'UMI', 'ESH', 
                          'ZMB', 'ALA']}

# define constants
REF_YEAR = 2014 # reference year
RES_ARCSEC = 300 # resolution in arc seconds
    
# define paths
exp_str = SYSTEM_DIR.joinpath(
    f"litpop_{RES_ARCSEC:04n}as_{REF_YEAR:04n}_global.hdf5")
cent_str = SYSTEM_DIR.joinpath(f"centroids_{RES_ARCSEC:04n}as_global.hdf5")

dist_cst_lim = 1000000

######################### Define functions ###################################

def init_coastal_litpop(countries, res_arcsec=300, ref_year=REF_YEAR, 
                        dist_cst_lim=dist_cst_lim, lat_lim=70., save=True):
    
    """Initiates LitPop exposure of all provided countries within a defined 
    distance to coast and extent of lat, lon.

    Parameters:
        countries (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        res_arcsec (int)
        ref_year (int):
        dist_cst_lim (int):
        lat_lim (float):
        save (boolean):
        
    Returns:
        DataFrame, hazard.centroids.centr.Centroids
    """
    
    success = []
    fail = []
    print("-----------------Initiating LitPop--------------------")
    exp_litpop = LitPop()
    print("Initiating LitPop country per country:....")
    for cntry in countries:
        print("-------------------------" + cntry + "--------------------------") 
        exp_litpop_tmp = LitPop()
        try:
            exp_litpop_tmp.set_country(cntry, res_arcsec=res_arcsec, reference_year=ref_year)
            exp_litpop_tmp.set_geometry_points()
            exp_litpop_tmp.set_lat_lon()
            try:
                reg_ids = np.unique(exp_litpop_tmp.region_id).tolist()
                dist_cst = dist_to_coast(np.array(exp_litpop_tmp.latitude), lon=np.array(exp_litpop_tmp.longitude))
                exp_litpop_tmp['dist_cst'] = dist_cst
                exp_litpop_tmp.loc[dist_cst > dist_cst_lim, 'region_id'] = -99
                exp_litpop_tmp = exp_litpop_tmp.loc[exp_litpop_tmp['region_id'].isin(reg_ids)]
                # exp_coast.plot_raster()
            except ValueError:
                print(cntry + ': distance to coast failed, exposure not trimmed')
            exp_litpop = exp_litpop.append(exp_litpop_tmp)
            success.append(cntry)
        except Exception as e:
            fail.append(cntry)
            print("Error while initiating LitPop Exposure for " + cntry + ". ", e)
    del exp_litpop_tmp
    print("----------------------Done---------------------")
    exp_litpop = exp_litpop.reset_index(drop=True)
    rows, cols, ras_trans = pts_to_raster_meta((exp_litpop.longitude.min(), \
            exp_litpop.latitude.min(), exp_litpop.longitude.max(), exp_litpop.latitude.max()), \
            min(get_resolution(exp_litpop.latitude, exp_litpop.longitude)))
    exp_litpop.meta = {'width':cols, 'height':rows, 'crs':exp_litpop.crs, 'transform':ras_trans}
    exp_litpop.set_geometry_points()
    exp_litpop.set_lat_lon()
    
    reg_ids = np.unique(exp_litpop.region_id).tolist()
    if -99 in reg_ids: reg_ids.remove(-99)
    if -77 in reg_ids: reg_ids.remove(-77)
    print('reg_ids:', reg_ids)
    exp_litpop.check()
    try:
        dist_cst = dist_to_coast(np.array(exp_litpop.latitude), lon=np.array(exp_litpop.longitude))
        print(max(dist_cst))
        exp_litpop['dist_cst'] = dist_cst
        exp_litpop.loc[dist_cst > dist_cst_lim, 'region_id'] = -99
        exp_litpop.loc[exp_litpop.latitude>lat_lim, 'region_id'] = -99
        exp_litpop.loc[exp_litpop.latitude<-lat_lim, 'region_id'] = -99
        print('rejected: ', np.argwhere(exp_litpop.region_id==-99).size)
        print('antes select:', exp_litpop.size)
        exp_coast = exp_litpop.loc[exp_litpop['region_id'].isin(reg_ids)]
        print('despues select:', exp_coast.size)

    except ValueError:
        print('distance to coast failed, exposure not trimmed')
        exp_coast = exp_litpop
    with open(SYSTEM_DIR.joinpath('cntry_fail.txt'), "w") as output:
        output.write(str(fail))
    with open(SYSTEM_DIR.joinpath('cntry_success.txt'), "w") as output:
        output.write(str(success))
    if save:
        exp_coast.write_hdf5(exp_str)
    return exp_coast

def init_centroids_manual(bbox=[-66.5, 66.5, -179.5, 179.5], res_arcsec=3600, \
                           id_offset=1e9, on_land=False):
    """initiates centroids depeding on grid border points and resolution"""
    # number of centroids in lat and lon direction:
    n_lat = np.int(np.round((bbox[1]-bbox[0])*3600/res_arcsec))+1
    n_lon = np.int(np.round((bbox[3]-bbox[2])*3600/res_arcsec))+1
    
    cent = Centroids()
    mgrid= (np.mgrid[bbox[0] : bbox[1] : complex(0, n_lat), \
                           bbox[2] : bbox[3] : complex(0, n_lon)]). \
                  reshape(2, n_lat*n_lon).transpose()
    cent.set_lat_lon(mgrid[:,0], mgrid[:,1])
    cent.set_on_land()
    if not on_land: # remove centroids on land
        cent = cent.select(sel_cen=~cent.on_land)
    cent.set_region_id()
    cent.check()
    return cent

def init_exposure(countries, make_plots=True, res_arcsec=RES_ARCSEC, ref_year=REF_YEAR):

    if Path.is_file(exp_str):
        print("----------------------Loading Exposure----------------------")
        exp_coast = LitPop.from_hdf5(exp_str) 
    else:
        print("----------------------Initiating Exposure-------------------")
        exp_coast = init_coastal_litpop(countries, res_arcsec=res_arcsec, 
                                        ref_year=ref_year, 
                                        dist_cst_lim=dist_cst_lim, lat_lim=70)
    return exp_coast


def init_centroids(exp):
    
    if Path.is_file(cent_str):
        print("----------------------Loading Exposure----------------------")
        cent = Centroids.from_hdf5(cent_str)
    else:
        cent.set_lat_lon(np.array(exp.latitude), np.array(exp.longitude.values))
        exp[INDICATOR_CENTR] = np.arange(cent.lat.size)
        cent.region_id = np.array(exp.region_id.values, dtype='int64')
        cent.on_land = np.ones(cent.lat.size)
        cent_sea = init_centroids_manual(id_offset=10**(1+len(str(int(cent.size)))), \
                                          res_arcsec=3600)
        cent.append(cent_sea)
        if np.unique(cent.coord, axis=0).size != 2*cent.coord.shape[0]:
            cent.remove_duplicate_points()
        cent.check()
    return cent

############################# Call functions #################################
# chose countries for exposure/centroids from Sam's calibration regions
cntry_list = []
# for global exposure and centroids
reg_list = ['NA1','NA2', 'NI', 'OC', 'SI', 'WP1', 'WP2', 'WP3', 'WP4', 'ROW']
for reg in reg_list:
    cntry_list.extend(region_ids_cal[reg])
    
exposure = init_exposure(cntry_list)
centroids = init_centroids(exposure)

# split exposure up for the four regions
# boundaries of (sub-)basins (lonmin, lonmax, latmin, latmax)
BASIN_BOUNDS = {
    # North Atlantic/Eastern Pacific Basin
    'AP': [-180.0, -30.0, 0.0, 65.0],

    # Indian Ocean Basin
    'IO': [30.0, 100.0, 0.0, 40.0],

    # Southern Hemisphere Basin
    'SH': [-180.0, 180.0, -60.0, 0.0],

    # Western Pacific Basin
    'WP': [100.0, 180.0, 0.0, 65.0],
}

for reg in ['AP', 'IO', 'SH', 'WP']:
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[str(reg)]
    exp_regional = LitPop()
    exp_regional.gdf = exposure.gdf.cx[x_min:x_max, y_min:y_max]
    exp_str = SYSTEM_DIR.joinpath(f"litpop_{RES_ARCSEC:04n}as_{REF_YEAR:04n}_{reg}.hdf5")
    exp_regional.write_hdf5(exp_str)