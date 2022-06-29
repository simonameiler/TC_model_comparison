"""
Adapted for code repository on 2022-06-28

description: Impact analysis - various analyses of impact results
            (from Impact_calc.py output)
            yields: Supplementary Tables 1-4, values for Hurricane Maria

@author: simonameiler
"""

import csv
import numpy as np
import scipy.stats as sp

# import CLIMADA modules:
from climada.engine import Impact

data_dir = "data"

basin_label = {'AP': 'North Atlantic/Eastern Pacific', 
               'IO': 'Indian Ocean', 
               'SH': 'Southern Hemisphere', 
               'WP': 'Western Pacific'}

def load_results(reg,load='IBTrACS_p'):
    load_str = data_dir/f"rp_imps_longCIs_{reg}_{load}.npz"
    with np.load(load_str) as fp:
        rp = fp['rp']
        imp_median = fp['imp_median']
        imp_q5 = fp['imp_q5']
        imp_q95 = fp['imp_q95']
    return rp, imp_median, imp_q5, imp_q95
       
def load_aai_agg(reg,load='IBTrACS_p'):
    load_str = data_dir/f"impacts_longCIs_{reg}_{load}.npz"
    with np.load(load_str) as fp:
        aai_agg = fp['aai_agg']
    return aai_agg

# to load the normalized impact results, just add "_norm" to the load string
# e.g. 'MIT_norm'
for reg in ['AP', 'IO', 'SH', 'WP']:
    
    rp, imp_median_i, imp_q5_i, imp_q95_i = load_results(reg, load='IBTrACS_p')
    _, imp_median_s, imp_q5_s, imp_q95_s = load_results(reg, load='STORM')
    _, imp_median_e, imp_q5_e, imp_q95_e = load_results(reg, load='MIT')
    _, imp_median_c, imp_q5_c, imp_q95_c = load_results(reg, load='CHAZ_ERA5')
    
    
    # combine mean, 5% and 95% frequencies in list
    median_list = [imp_median_i, imp_median_s, imp_median_e, imp_median_c]
    q5_list = [imp_q5_i, imp_q5_s, imp_q5_e, imp_q5_c]
    q95_list = [imp_q95_i, imp_q95_s, imp_q95_e, imp_q95_c]
    
    ##############################################################################
    # load estimated annual damage (EAD) = aai_agg variable in CLIMADA
    aai_agg = dict()
    aai_mean = []
    aai_median = []
    aai_std = []
    aai_iqr = []
    aai_iqr90 = []
    
    for aai in ['IBTrACS_p','STORM','MIT','CHAZ_ERA5']:
    # to load the "normalized" impact results, just add '_norm' to the load string
    #for aai in ['IBTrACS_p_norm','STORM_norm','MIT_norm','CHAZ_ERA5_norm']:
            aai_agg[aai] = load_aai_agg(reg,load=aai)
            aai_mean.append(np.mean(aai_agg[aai]))
            aai_median.append(np.median(aai_agg[aai]))
            aai_std.append(np.std(aai_agg[aai]))
            aai_iqr.append(sp.iqr(aai_agg[aai]))
            aai_iqr90.append(sp.iqr(aai_agg[aai],rng=[5,95]))
    
    
    # A) find the 100-yr and 1000-yr event - how?
    # 1) define where the RP exceeds the 100/1000 threshold the first time --> index
    # 2) read the corresponding impact from the impact lists
    
    ind_rp_100 = list(rp).index(rp[rp>100][0])
    ind_rp_1000 = list(rp).index(rp[rp>1000][0])
    
    imp_100_m = [median_list[i][ind_rp_100] for i in range(len(median_list))]
    imp_100_5 = [q5_list[i][ind_rp_100] for i in range(len(q5_list))]
    imp_100_95 = [q95_list[i][ind_rp_100] for i in range(len(q95_list))]
    imp_1000_m = [median_list[i][ind_rp_1000] for i in range(len(median_list))]
    imp_1000_5 = [q5_list[i][ind_rp_1000] for i in range(len(q5_list))]
    imp_1000_95 = [q95_list[i][ind_rp_1000] for i in range(len(q95_list))]
    
    header = ['aai mean', 'aai median', 'aai std', 'aai iqr', 'aai_iqr90',
              '100-yr median', '100-yr 5th', '100-yr 95th',
              '1000-yr median', '1000-yr 5th', '1000-yr 95th']
    data_matrix = np.zeros((4,11))
    for i in range(4):
        data_matrix[i,:] = [aai_mean[i], aai_median[i], aai_std[i], 
                            aai_iqr[i], aai_iqr90[i], imp_100_m[i], 
                            imp_100_5[i], imp_100_95[i],imp_1000_m[i], 
                            imp_1000_5[i], imp_1000_95[i]]
    
    # adapt string for "normalized" results f"aai_100_1000_event_{reg}_norm.csv"
    with open(data_dir.joinpath(f"aai_100_1000_event_{reg}.csv"), 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(data_matrix)

# EAD for the IBTrACS historical records
# change imp_str to f"TC_{reg}_impact_IBTrACS_norm.csv" for “normalized“ results
impact_dict = dict()
aai_dict_IB = dict()
for reg in ['AP', 'IO', 'SH', 'WP']:
    imp_str = f"TC_{reg}_impact_IBTrACS.csv"
    impact = Impact.from_csv(data_dir.joinpath(imp_str))
    impact_dict[reg] = impact
    aai_dict_IB[reg] = impact.aai_agg

####################### Hurricane Maria Analysis ##############################
reg = 'AP'
imp_str = f"TC_{reg}_impact_IBTrACS.csv"
impact_IBTrACS = Impact.from_csv(data_dir.joinpath(imp_str))

# IBTrACS event ID of Maria: 2017260N12310
maria_ind = impact_IBTrACS.event_name.index('2017260N12310')
maria_imp = impact_IBTrACS.at_event[maria_ind]

freq_curve = impact_IBTrACS.calc_freq_curve()

index, = np.where(freq_curve.impact > maria_imp)
rp_maria = freq_curve.return_per[index[0]]
freq_curve.return_per
freq_curve.return_per[-10:]

rp, imp_median_i, _, _ = load_results(reg, load='IBTrACS_p')
_, imp_median_s, _, _ = load_results(reg, load='STORM')
_, imp_median_e, _, _ = load_results(reg, load='MIT')
_, imp_median_c, _, _ = load_results(reg, load='CHAZ_ERA5')

# combine mean, 5% and 95% frequencies in list
median_list = [imp_median_i, imp_median_s, imp_median_e, imp_median_c]

maria_synth_ind = [list(imps).index(imps[imps>maria_imp][0]) for imps in median_list]
maria_synth_rp = [rp[i] for i in maria_synth_ind]
