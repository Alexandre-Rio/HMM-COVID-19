import warnings; warnings.simplefilter('ignore')  # hide warnings
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import particles
from particles import state_space_models as ssm
from particles.distributions import *

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
- *flight_info*: 1 if one or several passengers have been tested positive in evacuation flights. 0 otherwise.
All countries considered.
- *flight_prop*: First column indicates the number of passengers that have been tested positive in evacuation flights. 
Second column indicates the total number of passengers traveling in those evacuation flights. This allows to compute 
the proportion of passengers on evacuation flights that tested positive for SARS-CoV-2. All countries considered.
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

'''
Format data
'''

# Process flight prop (take only first column)
flight_prop_processed = flight_prop.to_numpy()
flight_prop_processed = flight_prop_processed[:, 0]

# Process international confirmed cases data from 20 most at-risk countries
relative_risk = pd.read_csv('data/connectivity_data_mobs.csv', encoding='cp1252').to_numpy()
top_20_countries = relative_risk[:20, 0]
int_case_conf = int_case_conf.to_numpy()
list_data = [int_case_conf[:, j] for j in range(20)]
int_case_conf_dict = dict(zip(top_20_countries, list_data))

# Define a general function to format data for filtering algorithms
def format_data(data_dict: dict):
    '''
    Format data for filtering algorithms
    :param data_dict: a dictionary containing the data to be fitted on
    :return: formatted data for filtering
    '''

    data = []
    time_range = len(data_dict[list(data_dict.keys())[0]])

    for t in range(time_range):
        data_t = OrderedDict()
        for field in data_dict.keys():
            data_dict[field][np.isnan(data_dict[field])] = 0
            data_t[field] = Dirac(loc=data_dict[field][t])
        data_t = StructDist(data_t)
        data.append(data_t.rvs())

    return data










