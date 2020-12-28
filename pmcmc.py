# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import pdb
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import *
from particles.mcmc import PMMH
import particles.distributions as distrib  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from model import *
from collections import OrderedDict
from data_processing import format_data, int_case_conf_dict, flight_prop_processed

local_case_data_onset = pd.read_csv('data/local_case_data_onset.csv', encoding='cp1252')
local_case_data_conf = pd.read_csv('data/local_case_data_conf.csv', encoding='cp1252')
int_case_onset = pd.read_csv('data/int_case_onset.csv', encoding='cp1252')
int_case_conf = int_case_conf_dict
flight_prop = flight_prop_processed
time_range = 82
use_validation = True

# Organize data in a dict
data_dict = {'onset': local_case_data_onset.to_numpy(),
           'onset_int': int_case_onset.to_numpy(),
           'reported': local_case_data_conf.to_numpy()
           }
if True:
    data_dict['flight_int'] = flight_prop
    for country in int_case_conf.keys():
        data_dict['reported_int_' + country] = int_case_conf[country]

# Check shapes
for field in data_dict.keys():
    assert data_dict[field].shape[0] == time_range

# Format data
data = format_data(data_dict)


# Number of flight passengers from flight_prop (for validation)
flight_prop = pd.read_csv('data/flight_prop.csv', encoding='cp1252')
flight_passengers = flight_prop['flight_prop.2'].values
flight_passengers[np.isnan(flight_passengers)] = 0  # Fill NaN values

# Relative risk of exporting cases (per country)
relative_risk = pd.read_csv('data/connectivity_data_mobs.csv', encoding='cp1252')
relative_risk = relative_risk.iloc[:20].to_numpy()
relative_risk = dict(zip(relative_risk[:, 0], relative_risk[:, 1]))


# Define Erlang Distribution class
class Erlang(distrib.LocScaleDist):
    def rvs(self, size=None):
        return erlang.rvs(a=1, loc=self.loc, scale=self.scale,
                          size=self.shape(size))
    def logpdf(self, x):
        return erlang.logpdf(x, a=1, loc=self.loc, scale=self.scale)
    def ppf(self, u):
        return erlang.ppf(u, a=1, loc=self.loc, scale=self.scale)


class Exponential(distrib.LocScaleDist):
    def rvs(self, size=None):
        return expon.rvs(loc=self.loc, scale=self.scale,
                         size=self.shape(size))
    def logpdf(self, x):
        return expon.logpdf(x, loc=self.loc, scale=self.scale)
    def ppf(self, u):
        return expon.ppf(u, loc=self.loc, scale=self.scale)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--nx', type=int, default=2e3, help='Number of X particles')
    args = parser.parse_args()

    def prior_distributions():
        prior = OrderedDict()
        prior['a'] = distrib.Normal(0.395, 1)
        prior['sigma'] = Erlang(scale=5.2)
        prior['kappa'] = Exponential(scale=6.1)
        prior['gamma'] = Erlang(scale=2.9)
        prior['delta'] = distrib.Uniform(0, 1)
        prior['pw'] = distrib.Uniform(0, 1)
        prior['pt'] = distrib.Uniform(0, 1)
        return distrib.StructDist(prior)

    prior = prior_distributions()


    pmmh = PMMH(niter=args.n_iter,
                ssm_cls=TransmissionModelExtended,
                prior=prior,  # StructDist
                data=data,  # list-like
                fk_cls=ssm.Bootstrap,
                Nx=args.nx,
                verbose=1
                # theta0=theta  # structured array of length 1
    )

    pdb.set_trace()

    pmmh.run()

    pdb.set_trace()





