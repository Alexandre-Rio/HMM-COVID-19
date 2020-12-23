import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import particles

__all__ = ['local_case_data_onset', 'local_case_data_conf', 'local_case_data_onset_scale', 'int_case_onset',
           'int_case_onset_scale', 'int_case_conf', 'flight_info', 'flight_prop', 'travel_data', 'date_range']

# Set path
directory = os.getcwd()
# sys.path.append(directory)
os.chdir(directory)

#from load_raw_data import *
# from process_data import *

"""## Description

Implementation of the paper **'Early dynamics of transmission and control of COVID-19: a mathematical
modelling study'**, based on the code provided in the git: https://github.com/adamkucharski/2020-ncov/tree/master/stoch_model_V2_paper

## Visualize data
"""

data_hubei_Feb = pd.read_csv('data/hubei_confirmed_cases.csv', index_col=0, encoding='cp1252')
travel_data = pd.read_csv('data/connectivity_data_mobs.csv', index_col=0, encoding='cp1252')

# Set tup start/end dates
new_date_hubei = data_hubei_Feb['date'].max()
start_date = pd.to_datetime('2019-11-22') # First case
end_date = new_date_hubei  # Period to forecast ahead
date_range = pd.date_range(start_date, end_date)

# Total cases in Wuhan from 15/01/2020 to 11/02/2020
df_Wuhan = data_hubei_Feb.loc[420100]
total_cases_Wuhan = pd.Series(df_Wuhan.total_case.values, index=df_Wuhan.date)
ax = total_cases_Wuhan.plot(title='Total number of cases in Wuhan', figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Cases')
#plt.show()

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

# Format data
local_case_data_onset.index = date_range
local_case_data_conf.index = date_range
local_case_data_onset_scale.index = date_range
int_case_onset.index = date_range

int_case_conf.index = date_range
int_case_conf.columns = travel_data.index

int_case_onset_scale.index = date_range
flight_info.index = date_range

flight_prop.index = date_range
flight_prop.columns = ['cases', 'nb_passengers']

"""I detected errors in the *int_case_conf* time series of Australia and USA. Indeed, for these countries, cases are reported by states. When cases are reported on the same day in two or more different states, only one report is taken into account. I fix these errors below."""

# Correct errors for Australia and USA
int_case_conf.loc['2020-01-25', 'Australia'] = 3
int_case_conf.loc['2020-01-29', 'Australia'] = 2
int_case_conf.loc['2020-01-26', 'USA'] = 3

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(local_case_data_onset, 'bx-')
axes[0, 0].set_title("Confirmed cases in China by date of onset")
axes[0, 0].set_ylabel("Number of cases")
axes[0, 1].plot(int_case_onset, 'bx-')
axes[0, 1].set_title("International cases by date of onset")
axes[0, 1].set_ylabel("Number of cases")
axes[1, 0].plot(flight_prop['cases']/flight_prop['nb_passengers'], 'ro')
axes[1, 0].set_title("Proportion of infected passengers in evacuation flights")
axes[1, 0].set_ylabel("Proportion of infected passengers")
axes[1, 1].plot(local_case_data_conf, 'bx-')
axes[1, 1].set_title("Confirmed cases in Wuhan by date of confirmation")
axes[1, 1].set_ylabel("Number of cases")

int_case_conf.sum(axis=1).plot(figsize=(14, 6))
plt.ylabel('Number of cases')
plt.title("International cases by date of confirmation (all countries)")
#plt.show()








