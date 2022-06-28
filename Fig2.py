"""
Adapted for code repository on 2022-06-28

description: Impact analysis - plotting impact return period curves

@author: simonameiler
"""

import numpy as np
import matplotlib.pyplot as plt

# import CLIMADA modules:
from climada.engine import Impact

# Define regions and paths and load impacts
data_dir = "data"

basin_label = {'AP': 'North Atlantic/Eastern Pacific', 
               'IO': 'North Indian Ocean', 
               'SH': 'Southern Hemisphere', 
               'WP': 'Western Pacific'}
    
basin_axes = {'AP': [0,0], 
              'IO': [0,1], 
              'SH': [1,0], 
              'WP': [1,1]}

colors = {'IBTrACS': 'k',
          'IBTrACS_p': '#3e3f40', 
          'STORM': '#b7222b', 
          'MIT': '#1116cb', 
          'CHAZ_ERA5': '#2abdc7'}

subplot_handles = ['a)', 'b)', 'c)', 'd)']

def load_results(reg,load='IBTrACS_p'):
    load_str = data_dir/f"rp_imps_longCIs_{reg}_{load}.npz"
    with np.load(load_str) as fp:
        rp = fp['rp']
        imp_median = fp['imp_median']
        imp_q5 = fp['imp_q5']
        imp_q95 = fp['imp_q95']
    return rp, imp_median, imp_q5, imp_q95

# load IBTrACS impact from csv and get impacts and return periods thereof
impact_dict = dict()
ibtracs_rp = dict()
ibtracs_imp = dict()
for reg in basin_label.keys():
    imp_str = f"TC_{reg}_impact_IBTrACS.csv"
    impact = Impact.from_csv(data_dir.joinpath(imp_str))
    impact_dict[reg] = impact
    # store impact and RP for every region in dict to plot later
    freq_curves = impact_dict[reg].calc_freq_curve()
    ibtracs_rp[reg] = freq_curves.return_per
    ibtracs_imp[reg] = freq_curves.impact

# plot results
fig, axis = plt.subplots(
    2,2, figsize=(12,8), tight_layout=True, sharex=True, sharey=True)
plt.rcParams.update({'font.size': 16})
basin_axes = {'AP': [0,0], 
              'IO': [0,1], 
              'SH': [1,0], 
              'WP': [1,1]}
for k, reg in enumerate(basin_label.keys()):
    rp, imp_median_i, imp_q5_i, imp_q95_i = load_results(reg, load='IBTrACS_p')
    rp, imp_median_s, imp_q5_s, imp_q95_s = load_results(reg, load='STORM')
    rp, imp_median_e, imp_q5_e, imp_q95_e = load_results(reg, load='MIT')
    rp, imp_median_c, imp_q5_c, imp_q95_c = load_results(reg, load='CHAZ_ERA5')
    axis[basin_axes[reg][0],basin_axes[reg][1]].plot(ibtracs_rp[reg], ibtracs_imp[reg],color='k', lw=2.0)
    axis[basin_axes[reg][0],basin_axes[reg][1]].plot(rp, imp_median_i,color='#3e3f40', lw=2.0)
    axis[basin_axes[reg][0],basin_axes[reg][1]].fill_between(rp,imp_q5_i,imp_q95_i, color='#3e3f40', alpha=0.3)
    axis[basin_axes[reg][0],basin_axes[reg][1]].plot(rp,imp_median_s,color='#b7222b', lw=2.0)
    axis[basin_axes[reg][0],basin_axes[reg][1]].fill_between(rp, imp_q5_s, imp_q95_s, color='#b7222b', alpha=0.3)
    axis[basin_axes[reg][0],basin_axes[reg][1]].plot(rp,imp_median_e,color='#1116cb', lw=2.0)
    axis[basin_axes[reg][0],basin_axes[reg][1]].fill_between(rp, imp_q5_e, imp_q95_e, color='#1116cb', alpha=0.3)
    axis[basin_axes[reg][0],basin_axes[reg][1]].plot(rp,imp_median_c,color='#2abdc7', lw=2.0)
    axis[basin_axes[reg][0],basin_axes[reg][1]].fill_between(rp, imp_q5_c, imp_q95_c, color='#2abdc7', alpha=0.3)
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_xscale('log')
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_yscale('log')
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_title(basin_label[reg])
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_ylim(1e6, 1.7e12)
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_xlim(7e-2,1e3)
    axis[basin_axes[reg][0],basin_axes[reg][1]].text(
        -0.1, 1.05, str(subplot_handles[k]), 
        transform=axis[basin_axes[reg][0],basin_axes[reg][1]].transAxes)
fig.legend(['IBTrACS','IBTrACS_p','STORM','MIT','CHAZ'],bbox_to_anchor=(1.15, 0.65))
fig.supylabel('Impact (USD)', fontsize=16)
fig.supxlabel('Return period (years)', fontsize=16, x=0.55)









