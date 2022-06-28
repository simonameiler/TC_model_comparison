"""
Adapted for code repository on 2022-06-28

description: Impact analysis - Tail risk assessment

@author: simonameiler
"""

import numpy as np
import matplotlib.pyplot as plt

# Define constants, paths

data_dir = "data"

basin_label = {'AP': 'North Atlantic/Eastern Pacific', 
               'IO': 'Indian Ocean', 
               'SH': 'Southern Hemisphere', 
               'WP': 'Western Pacific'}

fig, axis = plt.subplots(
    2,2, figsize=(12,8), tight_layout=True, sharex=True, sharey=True)
plt.rcParams.update({'font.size': 16})
basin_axes = {'AP': [0,0], 
              'IO': [0,1], 
              'SH': [1,0], 
              'WP': [1,1]}

def load_impact(reg,load='IBTrACS_p'):
    load_str = data_dir/f"impacts_longCIs_{reg}_{load}.npz"
    with np.load(load_str) as fp:
        impact = fp['impact']
        imp_size = fp['imp_size']
    return impact, imp_size

for k, reg in enumerate(basin_label.keys()):
    
    tail_impacts = dict()
    imp_size = dict()
    for tail in ['IBTrACS_p','STORM','MIT','CHAZ_ERA5']:
        tail_impacts[tail], imp_size[tail] = load_impact(reg,load=tail)
    
    plt.rcParams.update({'font.size': 15})
    colors = ['#3e3f40', '#b7222b', '#1116cb', '#2abdc7','g']
    subplot_handles = ['a)', 'b)', 'c)', 'd)']
    vp = axis[basin_axes[reg][0],basin_axes[reg][1]].violinplot(tail_impacts.values(), 
                       showmeans=False, showmedians=False, showextrema=False)
    len_colors = len(colors)
    i = 0
    for pc in vp['bodies']:
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        i += 1
        if i == len_colors:
            i = 0
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_xticks([1,2,3,4])
    for i, tl in enumerate(tail_impacts):
        axis[basin_axes[reg][0],basin_axes[reg][1]].text(
            i+0.8, 1.5e12, '{:.3%}'.format(tail_impacts[tl].size/imp_size[tl]))
        axis[basin_axes[reg][0],basin_axes[reg][1]].text(
            -0.1, 1.05, str(subplot_handles[k]), transform=axis[basin_axes[reg][0],basin_axes[reg][1]].transAxes)
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_xticklabels(['IBTrACS_p','STORM','MIT','CHAZ'])
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_yscale('log')
    axis[basin_axes[reg][0],basin_axes[reg][1]].set_title(basin_label[reg], fontsize=15)
axis[0,0].set_ylabel('Impact (USD)', fontsize=15)
axis[1,0].set_ylabel('Impact (USD)', fontsize=15)
