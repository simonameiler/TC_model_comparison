# Intercomparison of regional loss estimates from global synthetic tropcial cyclone models
These scripts reproduce the main results of the paper:

*Simona Meiler(1,2), Thomas Vogt(3), Nadia Bloemendaal(4), Alessio Ciullo(1,2), Chia-Ying Lee(5), Suzana J. Camargo(5),
Kerry Emanuel(6), and David N. Bresch(1,2):
**Intercomparison of regional loss estimates from global synthetic tropcial cyclone models***

Publication status: [under revision](https://doi.org/10.21203/rs.3.rs-1429968/v1).

(1) Institute for Environmental Decisions, ETH Zurich, Switzerland

(2) Federal Office of Meteorology and Climatology MeteoSwiss, Switzerland

(3) Potsdam Institute for Climate Impact Research (PIK), Potsdam, Germany

(4) Institute for Environmental Studies (IVM), Vrije Universiteit Amsterdam, Amsterdam, The Netherlands

(5) Lamont-Doherty Earth Observatory, Columbia University, Palisades, New York, USA

(6) Lorenz Center, Massachusetts Institute of Technology, Cambridge, Massachusetts, USA

Contact: [Simona Meiler](simona.meiler@usys.ethz.ch)


## Content:

#### Cent_exposure.py
Python script to generate the centroids and exposure files.

#### Hazard_*.py
Python scripts to load TC track sets and calculate the 2D windfield. The * stands for the TC track set used for each
respective calculation. The output hdf5 files are the hazard sets, which are further used for the impact calculation.
Note that this step requires a computer cluster and that the output files are large (up to over 10GB per file).

#### TC_subsample.py
Python script for hazard analysis: draw subsamples at length of IBTrACS (39 years).
Make sure to run this before running the remaining `TC_*.py` scripts.

#### TC_max_wind.py
Python script for hazard analysis: extract max wind speeds over land (!) from wind field data sets.
Make sure to run this before running the remaining `TC_*.py` scripts below.

#### TC_track_stats.py
Python script for hazard analysis: landfall intensity statistics from track data.
Make sure to run this after `TC_max_wind.py` and `TC_subsample.py`, and before running
the `TC_histograms.py` script below.

#### TC_haz_stats.py
Python script for hazard analysis: landfall intensity statistics from windfield data.
Make sure to run this after `TC_max_wind.py` and `TC_subsample.py`, and before running
the `TC_histograms.py` script below.

#### TC_histograms.py
Python script for hazard analysis: plot landfall intensity histograms, from track (Supplementary Figure 1) or
windfield data (Figure 1).
Make sure to run the `TC_track_stats.py` and `TC_haz_stats.py` scripts (above) before running this script.

#### Impact_calc.py
Python script to compute the estimated loss from different synthetic tropical cyclone hazard sets - run on
the ETH Euler cluster. The output csv files are post-processed in Impact_post-process.py.

#### Impact_post-process.py
Python script for post-processing of impact results; subsampling, confidence intervals, etc.
The output npz files are used for plotting and uploaded in the data folder.

#### Impact_tables_res.py
Python script containing impact analyses. Produces numbers for Supplementary Tables 1-4 and Hurricane Maria analysis.
Can be executed with the data in the "data" folder.

#### Fig*.py
Python scripts named according to their Figure number in the publication can be used to reproduce the figures.

#### Supp_*.py
Python scripts starting with Supp_ are used to produce outputs and results for the Supplementary Material and contain
code analogous to their main text counterparts.

## Requirements
Requires:
* Python 3.8+ environment (best to use conda for CLIMADA repository)
* _CLIMADA_ repository version 3.1.2+:
        https://wcr.ethz.ch/research/climada.html
        https://github.com/CLIMADA-project/climada_python

## ETH cluster
Computationally demanding calculations were run on the [Euler cluster of ETH Zurich](https://scicomp.ethz.ch/wiki/Euler).

## Documentation:
Publication: submitted to **Nature Communications**

Documentation for CLIMADA is available on Read the Docs:
* [online (recommended)](https://climada-python.readthedocs.io/en/stable/)
* [PDF file](https://buildmedia.readthedocs.org/media/pdf/climada-python/stable/climada-python.pdf)

If script fails, revert CLIMADA version to release v3.1.2:
* from [GitHub](https://github.com/CLIMADA-project/climada_python/releases/tag/v3.1.2)

## History

Created on 28 June 2022

-----

www.wcr.ethz.ch
