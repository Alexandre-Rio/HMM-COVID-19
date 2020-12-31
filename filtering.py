import warnings; warnings.simplefilter('ignore')  # hide warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import particles
from particles import state_space_models as ssm
from model import *
from data_processing import format_data, int_case_conf_dict, flight_prop_processed
import pdb

##############################################
########### LOAD AND FORMAT DATA #############
##############################################

os.chdir(os.getcwd())  # Change directory to current working directory
time_range = 82
use_validation = True

# Load data
local_case_data_onset = pd.read_csv('data/local_case_data_onset.csv', encoding='cp1252')
local_case_data_conf = pd.read_csv('data/local_case_data_conf.csv', encoding='cp1252')
int_case_onset = pd.read_csv('data/int_case_onset.csv', encoding='cp1252')
int_case_conf = int_case_conf_dict
flight_prop = flight_prop_processed

# Organize data in a dict
data_dict = {'onset': local_case_data_onset.to_numpy(),
           'onset_int': int_case_onset.to_numpy(),
           'reported': local_case_data_conf.to_numpy()
           }

if use_validation:
    data_dict['flight_int'] = flight_prop
    for country in int_case_conf.keys():
        data_dict['reported_int_' + country] = int_case_conf[country]

# Check shapes
for field in data_dict.keys():
    assert data_dict[field].shape[0] == time_range

# Format data
data = format_data(data_dict)


#####################################################
############## FILTERING ############################
#####################################################

# Define model
model = TransmissionModelExtended(**theta)

# Define Feyman-Kac model
fk_model = ssm.Bootstrap(ssm=model, data=data)

if __name__ == '__main__':

    # Define SMC algorithm parameters
    n_particles = 2000  # Number of particles
    n_runs = 200  # Number of runs
    resampling_mtd = 'multinomial'

    # Run algorithm n_runs times / for one run, just do "pf.run()"
    samples = []
    for run in range(n_runs):
        print(run)
        # Define Particle Filtering algorithm
        pf = particles.SMC(fk=fk_model, N=n_particles, resampling=resampling_mtd, store_history=True, summaries=False)
        # Run algorithm
        # pdb.set_trace()

        pf.run()
        # Sample latent variables from SMC-estimated distribution
        sample = pf.hist.backward_sampling(1)
        samples.append(np.array(sample))
    if n_runs == 1:
        samples = samples[0]
    samples = np.array(samples)


    # Plot estimated R0
    x = np.arange(time_range)
    r0 = samples['beta'] * theta['gamma']
    r0_mean = r0.mean(axis=0)
    r0_std = r0.std(axis=0)
    r0_ci = 1.96 * r0_std / np.sqrt(n_runs)
    plt.figure()
    plt.title('R0 - Estimated by SMC ({} runs, {} particles)'.format(n_runs, n_particles))
    plt.plot(x, r0_mean, color='royalblue')
    plt.fill_between(x, r0_mean - r0_ci, r0_mean + r0_ci, color='lightblue')
    plt.xlabel('Time steps')
    plt.ylabel('Estimated R0')
    plt.show()


end=True



