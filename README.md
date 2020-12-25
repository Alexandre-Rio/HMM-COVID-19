# Description

Implementation of the paper **'Early dynamics of transmission and control of COVID-19: a mathematical
modelling study'**, using the `particles` package.
Uses code from `https://github.com/adamkucharski/2020-ncov`.

# Processed data

Processed data comes from R script `2020-ncov/stoch_model_V2_paper/R/load_timeseries_data.R`.

Eventually, I found it more convenient to directly use the outputs of this script. It may be all the data we'll need.
So I saved in `.csv` files, whose content is described below.

**Descriptions:**
- *local_case_data_onset*: Confirmed cases in China by date of onset 
- *local_case_data_conf*: Confirmed cases in Wuhan by date of confirmation
- *local_case_data_onset_scale*: 
- *int_case_onset*: International cases by date of onset
- *int_case_conf*: International cases by date of confirmation of case
- *int_case_onset_scale*:

Information about evacuation flights:
- *flight_info*: 1 if one or several passengers have been tested positive in evacuation flights. 0 otherwise. All countries considered.
- *flight_prop*: First column indicates the number of passengers that have been tested positive in evacuation flights. Second column indicates the total number of passengers traveling in those evacuation flights. This allows to compute the proportion of passengers on evacuation flights that tested positive for SARS-CoV-2. All countries considered.

# State-Space Model

The State-Space Model described in the Appendix is implemented in `model.py`.

Model:
*To be written*

# SMC Filtering Algorithm

The SMC filtering algorithm is implemented in `model.py`.

# Other

`smc_model.py` is an adaptation of the R script `2020-ncov/stoch_model_V2_paper/R/smc_model.R`.
