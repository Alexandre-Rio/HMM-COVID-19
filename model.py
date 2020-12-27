import warnings; warnings.simplefilter('ignore')  # hide warnings
import os

# standard libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# modules from particles
import particles  # core module
from particles.distributions import *  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined

__all__ = ['theta', 'initial_values', 'initial_values_ext', 'flight_passengers',
           'TransmissionModel', 'TransmissionModelExtended']

# Set path
directory = os.getcwd()
# sys.path.append(directory)
os.chdir(directory)


##########################
##### SET PARAMETERS #####
##########################

# Fixed parameters as given in the Appendix
theta = {'brownian_vol': 0.395,  #a
         'pop_Wuhan': 1.10e7,  #N
         'incubation': 1 / 5.2,  #sigma
         'report': 1 / 6.1,  #kappa
         'recover': 1 / 2.9,  #gamma
         'initial_r0': 2.5,
         'passengers': 3300,
         'travel_prop': None,  #f
         'reported_prop': 1,  #omega
         'relative_report': 0.0066,  #delta
         'local_known_onsets': 0.16,  #pw
         'int_known_onsets': 0.47, #pt
         'travel_restriction': 63  #day on which travel restrictions have been put in place
        }
theta['travel_prop'] = theta['passengers'] / theta['pop_Wuhan']

# Initial values for local simplified model
initial_values = {'beta': theta['initial_r0'] * theta['recover'],
                  'sus': theta['pop_Wuhan'] - 1,
                  'exp_1': 0,
                  'exp_2': 0,
                  'inf_1': 1/2,
                  'inf_2': 1/2,
                  'Q': 0,
                  'D': 0,
                  'C': 0
                 }

# Initial values for extended model
initial_values_ext = {'beta': theta['initial_r0'] * theta['recover'],
                  'sus': theta['pop_Wuhan'] - 1,
                  'exp_1': 0,
                  'exp_2': 0,
                  'exp_1_int': 0,
                  'exp_2_int': 0,
                  'inf_1': 1/2,
                  'inf_2': 1/2,
                  'Q': 0,
                  'Q_int': 0,
                  'D': 0,
                  'D_int':0,
                  'C': 0,
                  'C_int':0
                 }

# Number of flight passengers from flight_prop (for validation)
flight_prop = pd.read_csv('data/flight_prop.csv', encoding='cp1252')
flight_passengers = flight_prop['flight_prop.2'].values
flight_passengers[np.isnan(flight_passengers)] = 0  # Fill NaN values

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

class TransmissionModel(ssm.StateSpaceModel):

    def PX0(self):  # Distribution of X_0

        init_var = OrderedDict()
        init_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=self.initial_values['beta'])
        init_var['sus'] = Dirac(loc=self.initial_values['sus'])
        init_var['exp_1'] = Dirac(loc=self.initial_values['exp_1'])
        init_var['exp_2'] = Dirac(loc=self.initial_values['exp_2'])
        init_var['inf_1'] = Dirac(loc=self.initial_values['inf_1'])
        init_var['inf_2'] = Dirac(loc=self.initial_values['inf_2'])
        init_var['Q'] = Dirac(loc=self.initial_values['Q'])
        init_var['D'] = Dirac(loc=self.initial_values['D'])
        init_var['C'] = Dirac(loc=self.initial_values['C'])

        init_dist = StructDist(init_var)

        return init_dist

    # Define model transition functions
    def sus(self, xp):
        return np.maximum(xp['sus'] - xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N, 0)

    def exp_1(self, t, xp):
        return xp['exp_1'] + \
               (1 - (t < self.travel_restriction) * self.f) * \
               xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N - 2 * self.sigma * xp['exp_1']

    def exp_2(self, xp):
        return xp['exp_2'] + 2 * self.sigma * xp['exp_1'] - 2 * self.sigma * xp['exp_2']

    def inf_1(self, xp):
        return xp['inf_1'] + 2 * self.sigma * xp['exp_2'] - 2 * self.gamma * xp['inf_1']

    def inf_2(self, xp):
        return xp['inf_2'] + 2 * self.gamma * xp['inf_1'] - 2 * self.gamma * xp['inf_2']

    def Q(self, xp):
        factor = np.exp(- self.gamma * self.kappa)
        return xp['Q'] + 2 * self.sigma * xp['exp_2'] * factor - self.kappa * xp['Q']

    def D(self, xp):
        factor = np.exp(- self.gamma * self.kappa)
        return xp['D'] + 2 * self.sigma * xp['exp_2'] * factor

    def C(self, xp):
        return xp['C'] + self.kappa * xp['Q']

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)

        lat_var = OrderedDict()
        lat_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=xp['beta'])
        lat_var['sus'] = Dirac(loc=self.sus(xp))
        lat_var['exp_1'] = Dirac(loc=self.exp_1(t, xp))
        lat_var['exp_2'] = Dirac(loc=self.exp_2(xp))
        lat_var['inf_1'] = Dirac(loc=self.inf_1(xp))
        lat_var['inf_2'] = Dirac(loc=self.inf_2(xp))
        lat_var['Q'] = Dirac(loc=self.Q(xp))
        lat_var['D'] = Dirac(loc=self.D(xp))
        lat_var['C'] = Dirac(loc=self.C(xp))

        trans_dist = StructDist(lat_var)

        return trans_dist

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)

        expected_obs = OrderedDict()
        if t == 0:
            expected_obs['onset'] = Poisson(rate=x['D'] * self.omega * self.delta * self.pw)
            expected_obs['reported'] = Poisson(rate=x['C'] * self.omega * self.delta)
        elif t > 0:
            expected_obs['onset'] = Poisson(rate=np.maximum(x['D'] - xp['D'], 0) * self.omega * self.delta * self.pw)
            expected_obs['reported'] = Poisson(rate=np.maximum(x['C'] - xp['C'], 0) * self.omega * self.delta)

        obs_dist = StructDist(expected_obs)

        return obs_dist

class TransmissionModelExtended(TransmissionModel):

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

    # Define model transition functions for international travelers
    def exp_1_int(self, t, xp):
        return xp['exp_1_int'] + \
               (t < self.travel_restriction) * self.f * \
               xp['beta'] * xp['sus'] * (xp['inf_1'] + xp['inf_2']) / self.N - 2 * self.sigma * xp['exp_1_int']

    def exp_2_int(self, xp):
        return xp['exp_2_int'] + 2 * self.sigma * xp['exp_1_int'] - 2 * self.sigma * xp['exp_2_int']

    def Q_int(self, xp):
        factor = np.exp(- self.gamma * self.kappa)
        return xp['Q_int'] + 2 * self.sigma * xp['exp_2_int'] * factor - self.kappa * xp['Q_int']

    def D_int(self, xp):
        factor = np.exp(- self.gamma * self.kappa)
        return xp['D_int'] + 2 * self.sigma * xp['exp_2_int'] * factor

    def C_int(self, xp):
        return xp['C_int'] + self.kappa * xp['Q_int']

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)

        lat_var = OrderedDict()
        lat_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=xp['beta'])
        lat_var['sus'] = Dirac(loc=self.sus(xp))
        lat_var['exp_1'] = Dirac(loc=self.exp_1(t, xp))
        lat_var['exp_2'] = Dirac(loc=self.exp_2(xp))
        lat_var['exp_1_int'] = Dirac(loc=self.exp_1_int(t, xp))
        lat_var['exp_2_int'] = Dirac(loc=self.exp_2_int(xp))
        lat_var['inf_1'] = Dirac(loc=self.inf_1(xp))
        lat_var['inf_2'] = Dirac(loc=self.inf_2(xp))
        lat_var['Q'] = Dirac(loc=self.Q(xp))
        lat_var['Q_int'] = Dirac(loc=self.Q_int(xp))
        lat_var['D'] = Dirac(loc=self.D(xp))
        lat_var['D_int'] = Dirac(loc=self.D_int(xp))
        lat_var['C'] = Dirac(loc=self.C(xp))
        lat_var['C_int'] = Dirac(loc=self.C_int(xp))

        trans_dist = StructDist(lat_var)

        return trans_dist

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)

        expected_obs = OrderedDict()
        if t == 0:
            expected_obs['onset'] = Poisson(rate=x['D'] * self.omega * self.delta * self.pw)
            expected_obs['onset_int'] = Poisson(rate=x['D_int']* self.omega * self.pt)
            expected_obs['reported'] = Poisson(rate=x['C'] * self.omega * self.delta)

        elif t > 0:
            expected_obs['onset'] = Poisson(rate=np.maximum(x['D'] - xp['D'], 0) * self.omega * self.delta * self.pw)
            expected_obs['onset_int'] = Poisson(rate=np.maximum(x['D_int'] - xp['D_int'], 0) * self.omega * self.pt)
            expected_obs['reported'] = Poisson(rate=np.maximum(x['C'] - xp['C'], 0) * self.omega * self.delta)


        if self.use_validation:
            expected_obs['flight_int'] = Binomial(n=self.flight_passengers[t],
                                            p=(x['exp_2'] + (1 - self.omega) * (x['inf_1'] + x['inf_2'])) / self.N)
            # expected_obs['reported_int'] = ...

        obs_dist = StructDist(expected_obs)

        return obs_dist

if __name__ == '__main__':

    # Simulate nb_simulations times from the model
    time_range = 82
    nb_simulations = 500

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