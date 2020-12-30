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
use_validation = False

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
model.use_validation = use_validation

# Define Feyman-Kac model
fk_model = ssm.Bootstrap(ssm=model, data=data)

if __name__ == '__main__':

    # Define SMC algorithm parameters
    n_particles = 2000  # Number of particles
    n_runs = 200  # Number of runs
    resampling_mtd = 'ssp'

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

    # Compute quantities of interest
    period_interest = np.arange(22, 82)  # 2019-12-14 to end date
    r0 = samples['beta'] / theta['gamma']
    r0_mean = r0.mean(axis=0)
    r0_std = r0.std(axis=0)
    r0_ci = 1.96 * r0_std / np.sqrt(n_runs)
    r0_quantiles = np.quantile(r0, q=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

    # Plot estimated R0 (mean and std over all runs)
    x = period_interest
    plt.figure()
    plt.title('R0 - Estimated by SMC ({} runs, {} particles)'.format(n_runs, n_particles))
    plt.plot(x, r0_mean[x], color='royalblue')
    plt.fill_between(x, r0_mean[x] - r0_ci[x], r0_mean[x] + r0_ci[x], color='lightblue')
    plt.ylim(0, 12)
    plt.xlabel('Time steps')
    plt.ylabel('Estimated R0')
    plt.show()

    # Plot estimated R0 (quantiles 0.025, 0.25, 0.5, 0.75, 0.975)
    plt.figure()
    plt.title('R0 - Estimated by SMC ({} runs, {} particles)'.format(n_runs, n_particles))
    plt.fill_between(x, r0_quantiles[0, period_interest], r0_quantiles[4, period_interest], color='lightblue')
    plt.fill_between(x, r0_quantiles[1, period_interest], r0_quantiles[3, period_interest], color='tab:blue')
    plt.plot(x, r0_quantiles[2, period_interest], color='black')
    plt.ylim(0, 12)
    plt.xlabel('Time steps')
    plt.ylabel('Estimated R0')
    plt.show()

end=True



