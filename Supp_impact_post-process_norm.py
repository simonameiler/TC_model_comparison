"""
Adapted for code repository on 2022-06-28

description: Impact analysis on normalized exposure layer - post-processing; 
            confidence intervals (from Supp_impact_calc_norm.py output)

@author: simonameiler
"""

import sys
import numpy as np
import datetime as dt
from scipy.io import loadmat

# import CLIMADA modules:
from climada.util.constants import SYSTEM_DIR
from climada.engine import Impact


def main(reg):

    impact_dir = SYSTEM_DIR/"impact"
    res_dir = SYSTEM_DIR/"results"
    
    freq_corr_CHAZ = {'AP': 1.0334,
                      'IO': 0.9175,
                      'SH': 0.8642,
                      'WP': 0.8039}
    
    impact_dict = dict()
    for imp in ['IBTrACS','IBTrACS_p','STORM','MIT','CHAZ_ERA5']:
        imp_str = f"TC_{reg}_impact_{imp}_norm.csv"
        impact = Impact.from_csv(impact_dir.joinpath(imp_str))
        impact_dict[imp] = impact
    
    # Exceedance probability curve and annual average impact
    freq_curves = dict()
    aai_agg = dict()
    
    for freq in impact_dict:
        freq_curves[freq] = impact_dict[freq].calc_freq_curve()
        aai_agg[freq] = impact_dict[freq].aai_agg
    
    ######################## Conficence intervals #############################
    
    # functions to derive sampling error as in TC_imp_analysis.py
    rng = np.random.default_rng(123456789)   
    
    # subsampling IBTrACS INCL. original record
    def draws_sampling_error_IBTrACS(impact_dict, yrs_hist=39):
        gen = list()
        for i in range(100):
            gen.append('gen'+str(i))
        
        event_gen = np.array(
            (int(impact_dict['IBTrACS_p'].at_event.size/len(gen)))*gen)
    
        impact_draws_IBTrACS = dict()
        for ensemble in np.unique(event_gen):
            event_selected = (event_gen == ensemble)
            new_impact = Impact()
            new_impact.at_event = impact_dict['IBTrACS_p'].at_event[event_selected]
            new_impact.frequency = np.ones(new_impact.at_event.size)*(1/yrs_hist)
            new_impact.aai_agg = sum(new_impact.at_event * new_impact.frequency)
            new_impact.unit = impact_dict['IBTrACS'].unit
            impact_draws_IBTrACS['IBTrACS_p_'+str(ensemble)] = new_impact
        return impact_draws_IBTrACS

    # subsampling STORM
    def draws_sampling_error_STORM(N=1000, yrs_hist=39):
        event_year = np.array([
           int(n.split(".")[0][-1]) * 1000 + int(n.split("-")[1])
           for n in impact_dict['STORM'].event_name])
    
        impact_draws_STORM = dict()
        for draw in range(N):
            draw_years = rng.choice(
                np.arange(10000), yrs_hist, replace=False)
            event_selected = np.isin(event_year, draw_years)
            new_impact = Impact()
            new_impact.at_event = impact_dict['STORM'].at_event[event_selected]
            new_impact.frequency = np.ones(new_impact.at_event.size)*(1/yrs_hist)
            new_impact.aai_agg = sum(new_impact.at_event * new_impact.frequency)
            new_impact.unit = impact_dict['IBTrACS'].unit
            impact_draws_STORM['STORM_'+str(draw)] = new_impact
        return impact_draws_STORM        

    # subsampling MIT
    def draws_sampling_error_MIT(N=1000, yrs_hist=39):
        # load list of freq_year values from matlab file
        hazard_dir = SYSTEM_DIR/"hazard"
        freq_year = []
        freq_year = loadmat(hazard_dir.joinpath(
            f"freqyear_{reg}.mat"))['freqyear'][0].tolist()
        
        impact_draws_MIT = dict()
        for draw in range(N):
            event_year = np.array([
                dt.datetime.fromordinal(d).year 
                for d in impact_dict['MIT'].date.astype(int)])
            
            year_sample_sizes = {
                year: n
                for year, n in zip(range(1979, 2020), rng.poisson(freq_year))}
        
            event_selected = np.concatenate([
                rng.choice((event_year == year).nonzero()[0], 
                                 size=year_sample_sizes[year],
                                 replace=False)
                for year in range(1980, 2019)])
            
            new_impact =  Impact()
            new_impact.at_event = impact_dict['MIT'].at_event[event_selected]
            new_impact.frequency = np.ones(new_impact.at_event.size)*(1/yrs_hist)
            new_impact.aai_agg = sum(new_impact.at_event * new_impact.frequency)
            new_impact.unit = impact_dict['IBTrACS'].unit
            impact_draws_MIT['MIT_'+str(draw)] = new_impact
        return impact_draws_MIT

    # subsampling CHAZ  
    def draws_sampling_error_CHAZ(yrs_hist=39): 
        event_ensemble = np.array([
            int(n[17:19]) * 100 + int(n.split("-")[2])
            for n in impact_dict['CHAZ_ERA5'].event_name])
        
        impact_draws_CHAZ = dict()
        for ensemble in np.unique(event_ensemble):
            event_selected = (event_ensemble == ensemble)
            new_impact = Impact()
            new_impact.at_event = impact_dict['CHAZ_ERA5'].at_event[event_selected]
            #new_impact.frequency = np.ones(new_impact.at_event.size)*(1/yrs_hist)
            bias_corr = freq_corr_CHAZ[reg]
            new_impact.frequency = np.full(
                new_impact.at_event.size, bias_corr) / yrs_hist
            new_impact.aai_agg = sum(new_impact.at_event * new_impact.frequency)
            new_impact.unit = impact_dict['IBTrACS'].unit
            impact_draws_CHAZ['CHAZ_ERA5_'+str(ensemble)] = new_impact
        return impact_draws_CHAZ
    
    # derive confidence interval (CI) of impacts
    def derive_CI_imp(imp_dict, N=1000, yrs_hist=39):
        freq_curves = dict()
        for fr in imp_dict:
            freq_curves[fr] = imp_dict[fr].calc_freq_curve(rp)
 
        imp_all = np.zeros((N,rp_num))
        for i,q in enumerate(freq_curves):
            imp_all[i,:] = freq_curves[q].impact
        
        imp_median = np.median(imp_all, axis=0)
        imp_q5 = np.percentile(imp_all, 5, axis=0)
        imp_q95 = np.percentile(imp_all, 95, axis=0)
        return imp_median, imp_q5, imp_q95
    
    def get_impact_at_event(imp_dict, N=1000, yrs_hist=39):
        aai_agg = []
        tail_events = []
        impact_size = []
        for sample in imp_dict:
            aai_agg.append(imp_dict[sample].aai_agg)
            tail_events.append(
                imp_dict[sample].at_event[imp_dict[sample].at_event> 1e11])
            impact_size.append((imp_dict[sample].at_event > 0).sum())
        new_tail = np.concatenate(tail_events, axis=0)
        imp_size = np.sum(impact_size)
        return aai_agg, new_tail, imp_size

    def chunks(data, size=26):
        dict_selected = np.concatenate([
            rng.choice(list(data.values()), size, replace=False)])
        event_array = []
        for s, sel in enumerate(dict_selected):
            event_array.extend(dict_selected[s].at_event)
        return np.array(event_array)
    
    def save_results_CIs(imp_dict, N=1000, yrs_hist=1014, save='IBTrACS_p'):
        imp_median, imp_q5, imp_q95 = derive_CI_imp(imp_dict, N=N, yrs_hist=yrs_hist)
        save_str = res_dir/f"rp_imps_longCIs_{reg}_{save}_norm.npz"
        np.savez_compressed(
            save_str, imp_median=imp_median, imp_q5=imp_q5, imp_q95=imp_q95, rp=rp)
        return print(f"Results of {save} saved")
    
    def save_results_impact(imp_dict, N=1000, yrs_hist=1014, save='IBTrACS_p'):
        aai_agg, impact, imp_size = get_impact_at_event(imp_dict, N=N, yrs_hist=yrs_hist)
        save_str = res_dir/f"impacts_longCIs_{reg}_{save}_norm.npz"
        np.savez_compressed(
            save_str, aai_agg=aai_agg, impact=impact, imp_size=imp_size)
        return print(f"Impacts of {save} saved")
    
    
    ############################ Call functions ##################################
    # get subsamples
    ibtracs = draws_sampling_error_IBTrACS(impact_dict, yrs_hist=39)
    storm = draws_sampling_error_STORM(N=1000,yrs_hist=1014)
    MIT = draws_sampling_error_MIT()  
    chaz = draws_sampling_error_CHAZ()
    
    # concat subsamples of 39 year length to approximately 1000 years (except STORM)
    long_ibtracs = dict()
    for n in range(1000):
        concat_IB = Impact()
        concat_IB.at_event = chunks(ibtracs, 26)
        concat_IB.unit = impact_dict['IBTrACS'].unit
        concat_IB.frequency = np.ones(concat_IB.at_event.size)*(1/1014)
        concat_IB.aai_agg = sum(concat_IB.at_event * concat_IB.frequency)
        long_ibtracs['MIT_long_'+str(n)] = concat_IB 
    
    long_MIT = dict()
    for n in range(1000):
        concat_MIT = Impact()
        concat_MIT.at_event = chunks(MIT, 26)
        concat_MIT.unit = impact_dict['IBTrACS'].unit
        concat_MIT.frequency = np.ones(concat_MIT.at_event.size)*(1/1014)
        concat_MIT.aai_agg = sum(concat_MIT.at_event * concat_MIT.frequency)
        long_MIT['MIT_long_'+str(n)] = concat_MIT 
    
    long_chaz = dict()
    for n in range(1000):
        concat_chaz = Impact()
        concat_chaz.at_event = chunks(chaz, 26)
        concat_chaz.unit = impact_dict['IBTrACS'].unit
        bias_corr = freq_corr_CHAZ[reg]
        concat_chaz.frequency = np.full(concat_chaz.at_event.size, bias_corr)/1014
        concat_chaz.aai_agg = sum(concat_chaz.at_event * concat_chaz.frequency)
        long_chaz['CHAZ_long_'+str(n)] = concat_chaz 
    
    
    # call function to derive CIs
    rp_num = max([curv.impact.size for curv in freq_curves.values()])
    rp = np.linspace(0, 1014, num=rp_num)
    
    save_results_impact(long_ibtracs, N=1000, yrs_hist=1014, save='IBTrACS_p')
    save_results_impact(storm, N=1000, yrs_hist=1014, save='STORM')
    save_results_impact(long_MIT, N=1000, yrs_hist=1014, save='MIT')
    save_results_impact(long_chaz, N=1000, yrs_hist=1014, save='CHAZ_ERA5')

    save_results_CIs(long_ibtracs, N=1000, yrs_hist=1014, save='IBTrACS_p')
    save_results_CIs(storm, N=1000, yrs_hist=1014, save='STORM')
    save_results_CIs(long_MIT, N=1000, yrs_hist=1014, save='MIT')
    save_results_CIs(long_chaz, N=1000, yrs_hist=1014, save='CHAZ_ERA5')

if __name__ == "__main__":
    main(*sys.argv[1:])