# Description

Implementation of the paper **'Early dynamics of transmission and control of COVID-19: a mathematical
modelling study'**, based on the code provided in the git: https://github.com/adamkucharski/2020-ncov/tree/master/stoch_model_V2_paper

# Raw data

Load raw data from the `data` folder with `load_raw_data.py`.

Dates originally in `str` format ('yyyy-mm-dd'). Converted in `datetime` format.

**Description of the datasets:**
- `travel_data_mobs`: Relative risk (probability) of exporting a local case from China to 121 foreign countries. Used to identify the top 20 most at risk.
- `travel_data_worldpop`: Not used.
- `international_conf_data_in`: Daily number of new exported cases from Wuhan (or lack thereof) in countries with high connectivity to Wuhan (i.e. top 20 most at risk), by date of confirmation, as of 10th February 2020 (data from 13/01/2020 to 05/02/2020).
- `international_onset_data_in`: Daily number of new internationally exported cases (or lack thereof), by date of onset, as of 26th January 2020 (data from 30/12/2019 to 26/01/2020).
- `china_onset_data_in`: Daily number of new cases in China, by date of onset, between 29th December 2019 and 23rd January 2020 (data from 29/12/2019 to 23/01/2020).
- `wuhan_onset_data_in`: Daily number of new cases in Wuhan, with market exposure, by date of onset, between 1st December 2019 and 1st January 2020 (data from 01/12/2019 to 02/01/2020).
- `wuhan_onset_2020_01_30`: Daily number of new cases in Wuhan, with market exposure, by date of onset, between 1st December 2019 and 1st January 2020 (data from 08/12/2019 to 21/01/2020). WARNING: Not exactly the same data as above.
- `wuhan_conf_data_in`: Data on new confirmed cases reported in Wuhan between 16th January and 11th February 2020 (data from 31/12/2019 to 25/01/2020).
- `data_hubei_Feb`: Total cases in the province of Hubei from 15/01/2020 to 11/02/2020 per county code (Wuhan: 420100).

# Processed data

Raw data is processed in `process_data.py` (unfinished). 
Adapted from R script `2020-ncov/stoch_model_V2_paper/R/load_timeseries_data.R`.

Eventually, I found it more convenient to directly use the outputs of this script. It may be all the data we'll need.
So I saved in `.csv` files, whose content is described below.

See `Workbook.ipynb` to find out how to load and format outputs from R script `load_timeseries_data.R`.

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

# Next steps

From the data described right above, implement the stochastic models described at the beginning of the Appendix.
