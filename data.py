import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import particles

__all__ = ['local_case_data_onset', 'local_case_data_conf', 'local_case_data_onset_scale', 'int_case_onset',
           'int_case_onset_scale', 'int_case_conf', 'flight_info', 'flight_prop']

# Set path
directory = os.getcwd()
# sys.path.append(directory)
os.chdir(directory)

"""## Load processed data

Load and format outputs from R script 'R/load_timeseries_data.R'.

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
"""

# Load data
local_case_data_onset = pd.read_csv('data/local_case_data_onset.csv', encoding='cp1252')
local_case_data_conf = pd.read_csv('data/local_case_data_conf.csv', encoding='cp1252')
local_case_data_onset_scale = pd.read_csv('data/local_case_data_onset_scale.csv', encoding='cp1252')
int_case_onset = pd.read_csv('data/int_case_onset.csv', encoding='cp1252')
int_case_conf = pd.read_csv('data/int_case_conf.csv', encoding='cp1252')
int_case_onset_scale = pd.read_csv('data/int_case_onset_scale.csv', encoding='cp1252')
flight_info = pd.read_csv('data/flight_info.csv', encoding='cp1252')
flight_prop = pd.read_csv('data/flight_prop.csv', encoding='cp1252')










