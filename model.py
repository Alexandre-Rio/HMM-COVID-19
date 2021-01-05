import warnings; warnings.simplefilter('ignore')  # hide warnings
import os

# standard libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pdb

# modules from particles
import particles  # core module
from particles.distributions import *  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined

__all__ = ['theta', 'TransmissionModelExtended']

# Set path
directory = os.getcwd()
# sys.path.append(directory)
os.chdir(directory)


##########################
##### SET PARAMETERS #####
##########################

# Fixed parameters as given in the Appendix
theta = {'a': 0.395,  # brownian_vol
         'N': int(1.10e7),  # pop_Wuhan
         'sigma': 5.2,  # incubation
         'kappa': 6.1,  # report
         'gamma': 2.9,  # recover
         'initial_r0': 2.5,
         'passengers': 3300,
         'f': None,  # travel_prop
         'omega': 1,  # reported_prop
         'delta': 0.0066,  # relative_report
         'pw': 0.16,  # local_known_onsets
         'pt': 0.47,  # int_known_onsets
         'travel_restriction': 63  #day on which travel restrictions have been put in place
        }
theta['f'] = theta['passengers'] / theta['N']


####################################
##### DEFINE STATE SPACE MODEL #####
####################################


class ExpD(TransformedDist):
    """Distribution of Y = exp(X).

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    """
    def f(self, x):
        return np.exp(x)

    def finv(self, x):
        return np.log(x)

    def logJac(self, x):
        return - np.log(x)



# Define model object

class TransmissionModelExtended(ssm.StateSpaceModel):
    def __init__(self, *args, **kwargs):
        super(TransmissionModelExtended, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Fixed args
        self.N = int(1.10e7)
        self.travel_restriction = 63
        self.initial_r0 = 2.5
        self.passengers = 3300
        self.f = self.passengers / self.N

        # inv
        self.gamma = 1 / self.gamma
        self.sigma = 1 / self.sigma
        self.kappa = 1 / self.kappa

        self.initial_values = {'beta': self.initial_r0 * self.gamma, 'sus': self.N - 1,
                               'exp_1': 0, 'exp_2': 0, 'exp_1_int': 0, 'exp_2_int': 0, 'inf_1': 1/2, 'inf_2': 1/2,
                               'Q': 0, 'Q_int': 0, 'D': 0, 'D_int': 0, 'C': 0, 'C_int': 0
                               }


        # Number of flight passengers from flight_prop (for validation)
        flight_prop = pd.read_csv('data/flight_prop.csv', encoding='cp1252')
        self.flight_passengers = flight_prop['flight_prop.2'].values
        self.flight_passengers[np.isnan(self.flight_passengers)] = 0  # Fill NaN values

        # Relative risk of exporting cases (per country)
        self.relative_risk = pd.read_csv('data/connectivity_data_mobs.csv', encoding='cp1252')
        self.relative_risk = self.relative_risk.iloc[:20].to_numpy()
        self.relative_risk = dict(zip(self.relative_risk[:, 0], self.relative_risk[:, 1]))

        self.use_validation = False  # True

    def PX0(self):  # Distribution of X_0

        init_var = OrderedDict()
        init_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=self.initial_values['beta'])
        init_var['sus'] = Dirac(loc=self.initial_values['sus'])
        init_var['exp_1'] = Dirac(loc=self.initial_values['exp_1'])
        init_var['exp_2'] = Dirac(loc=self.initial_values['exp_2'])
        init_var['exp_1_int'] = Dirac(loc=self.initial_values['exp_1_int'])
        init_var['exp_2_int'] = Dirac(loc=self.initial_values['exp_2_int'])
        init_var['inf_1'] = Dirac(loc=self.initial_values['inf_1'])
        init_var['inf_2'] = Dirac(loc=self.initial_values['inf_2'])
        init_var['Q'] = Dirac(loc=self.initial_values['Q'])
        init_var['Q_int'] = Dirac(loc=self.initial_values['Q_int'])
        init_var['D'] = Dirac(loc=self.initial_values['D'])
        init_var['D_int'] = Dirac(loc=self.initial_values['D_int'])
        init_var['C'] = Dirac(loc=self.initial_values['C'])
        init_var['C_int'] = Dirac(loc=self.initial_values['C_int'])

        init_dist = StructDist(init_var)

        return init_dist

    # Define model transition functions

    def update_x(self, t, xp):
        self.sus = np.maximum(xp['sus'] - xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N, 0).astype('int64')
        self.exp_1 = xp['exp_1'] + (1 - (t < self.travel_restriction) * self.f) * \
                xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N - 2 * self.sigma * xp['exp_1']
        self.exp_2 = xp['exp_2'] + 2 * self.sigma * xp['exp_1'] - 2 * self.sigma * xp['exp_2']
        self.inf_1 = xp['inf_1'] + 2 * self.sigma * xp['exp_2'] - 2 * self.gamma * xp['inf_1']
        self.inf_2 = xp['inf_2'] + 2 * self.gamma * xp['inf_1'] - 2 * self.gamma * xp['inf_2']
        self.Q = xp['Q'] + 2 * self.sigma * xp['exp_2'] * np.exp(- self.gamma * self.kappa) - self.kappa * xp['Q']
        self.D = xp['D'] + 2 * self.sigma * xp['exp_2'] * np.exp(- self.gamma * self.kappa)
        self.C = xp['C'] + self.kappa * xp['Q']
        self.exp_1_int = xp['exp_1_int'] + (t < self.travel_restriction) * self.f * \
                    xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N - 2 * self.sigma * xp['exp_1_int']
        self.exp_2_int = xp['exp_2_int'] + 2 * self.sigma * xp['exp_1_int'] - 2 * self.sigma * xp['exp_2_int']
        self.Q_int = xp['Q_int'] + 2 * self.sigma * xp['exp_2_int'] * np.exp(- self.gamma * self.kappa) - self.kappa * xp['Q_int']
        self.D_int = xp['D_int'] + 2 * self.sigma * xp['exp_2_int'] * np.exp(- self.gamma * self.kappa)
        self.C_int = xp['C_int'] + self.kappa * xp['Q_int']

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)

        self.update_x(t, xp)

        lat_var = OrderedDict()
        lat_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=xp['beta'])
        lat_var['sus'] = Dirac(loc=self.sus)
        lat_var['exp_1'] = Dirac(loc=self.exp_1)
        lat_var['exp_2'] = Dirac(loc=self.exp_2)
        lat_var['exp_1_int'] = Dirac(loc=self.exp_1_int)
        lat_var['exp_2_int'] = Dirac(loc=self.exp_2_int)
        lat_var['inf_1'] = Dirac(loc=self.inf_1)
        lat_var['inf_2'] = Dirac(loc=self.inf_2)
        lat_var['Q'] = Dirac(loc=self.Q)
        lat_var['Q_int'] = Dirac(loc=self.Q_int)
        lat_var['D'] = Dirac(loc=self.D)
        lat_var['D_int'] = Dirac(loc=self.D_int)
        lat_var['C'] = Dirac(loc=self.C)
        lat_var['C_int'] = Dirac(loc=self.C_int)

        trans_dist = StructDist(lat_var)

        return trans_dist

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)

        expected_obs = OrderedDict()
        if t == 0:
            expected_obs['onset'] = Poisson(rate=x['D'] * self.omega * self.delta * self.pw)
            expected_obs['onset_int'] = Poisson(rate=x['D_int'] * self.omega * self.pt)
            # expected_obs['reported'] = Poisson(rate=x['C'] * self.omega * self.delta)
            expected_obs['flight_int'] = Binomial(n=self.flight_passengers[t], p=(x['exp_1'] + x['exp_2'] + (1 - self.omega) *
                                                                                  (x['inf_1'] + x['inf_2'])) / self.N)  # x['exp_1']

        elif t > 0:
            expected_obs['onset'] = Poisson(rate=np.maximum(x['D'] - xp['D'], 0) * self.omega * self.delta * self.pw)
            expected_obs['onset_int'] = Poisson(rate=np.maximum(x['D_int'] - xp['D_int'], 0) * self.omega * self.pt)
            # expected_obs['reported'] = Poisson(rate=np.maximum(x['C'] - xp['C'], 0) * self.omega * self.delta)
            expected_obs['flight_int'] = Binomial(n=self.flight_passengers[t], p=(x['exp_1'] + x['exp_2'] + (1 - self.omega) *
                                                                                  (x['inf_1'] + x['inf_2'])) / self.N)  # x['exp_1']

        if self.use_validation:
            for country in self.relative_risk.keys():
                if t == 0:
                    expected_obs['reported_int_' + country] = Poisson(rate=x['C_int'] * self.relative_risk[country])
                elif t > 0:
                    expected_obs['reported_int_' + country] = Poisson(rate=(x['C_int'] - xp['C_int']) *
                                                                           self.relative_risk[country])

        obs_dist = StructDist(expected_obs)

        return obs_dist


if __name__ == '__main__':

    # Simulate nb_simulations times from the model
    time_range = 82
    nb_simulations = 1

    model = TransmissionModelExtended(**theta)

    hidden_states, observations = model.simulate(time_range)

    sim_states = []
    sim_data = []
    for sim in range(nb_simulations):
        print(sim)
        hidden_states, observations = model.simulate(time_range)
        for i in range(time_range):
            hidden_states[i] = [hidden_states[i][0][j] for j in range(len(hidden_states[i][0]))]
            observations[i] = [observations[i][0][j] for j in range(len(observations[i][0]))]
        sim_states.append(hidden_states)
        sim_data.append(observations)

    sim_states, sim_data = np.array(sim_states), np.array(sim_data)
    avg_sim_states = sim_states.mean(axis=0)
    avg_sim_data = sim_data.mean(axis=0)


    end = True