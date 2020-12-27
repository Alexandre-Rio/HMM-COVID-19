import warnings; warnings.simplefilter('ignore')  # hide warnings
import os
import pandas as pd
import numpy as np
import particles
from particles import state_space_models as ssm
from model import *

##############################################
########### LOAD AND FORMAT DATA #############
##############################################

os.chdir(os.getcwd())  # Change directory to current working directory
time_range = 82

# Load data
local_case_data_onset = pd.read_csv('data/local_case_data_onset.csv', encoding='cp1252')
local_case_data_conf = pd.read_csv('data/local_case_data_conf.csv', encoding='cp1252')
int_case_onset = pd.read_csv('data/int_case_onset.csv', encoding='cp1252')

dict_data={'onset': local_case_data_onset,
           'onset_int': int_case_onset,
           'reported': local_case_data_conf
           }

# Check shapes
for data in dict_data.keys():
    assert dict_data[data].shape[0] == time_range

# Format data
data = pd.concat([dict_data[data] for data in dict_data.keys()], axis=1).to_numpy()

#####################################################
############## FILTERING ############################
#####################################################

# Define model
model = TransmissionModelExtended(a=theta['brownian_vol'],
                              N=theta['pop_Wuhan'],
                              sigma=theta['incubation'],
                              kappa=theta['report'],
                              gamma=theta['recover'],
                              f=theta['travel_prop'],
                              omega=theta['reported_prop'],
                              delta=theta['relative_report'],
                              pw=theta['local_known_onsets'],
                              pt=theta['int_known_onsets'],
                              travel_restriction=theta['travel_restriction'],
                              initial_values=initial_values_ext,
                              flight_passengers=flight_passengers,
                              use_validation = False
                              )

# To use simulated data instead of real data
# latent_states, data = model.simulate(time_range)
# To fill NaN values with zeros
mask = np.isnan(data)
data[mask] = 0

# Define Feyman-Kac model
fk_model = ssm.Bootstrap(ssm=model, data=data)

# Define SMC filtering algorithm
pf = particles.SMC(fk=fk_model, N=10, resampling='stratified', moments=True, store_history=True, summaries=False)

# Run algorithm
next(pf)
pf.run()

end=True



