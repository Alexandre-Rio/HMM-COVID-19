import warnings; warnings.simplefilter('ignore')  # hide warnings

# standard libraries
from matplotlib import pyplot as plt
import numpy as np

# modules from particles
import particles  # core module
from particles.distributions import *  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined


##########################
##### SET PARAMETERS #####
##########################

theta = {'brownian_vol': 0.395,  #a
         'pop_Wuhan': 1.10e7,  #N
         'incubation': 1 / 5.2,  #sigma
         'report': 1 / 6.1,  #kappa
         'recover': 1 / 2.9,  #gamma
         'initial_r0': 2.5,
         'passengers': 3300,
         'travel_prop': None,  #f
         'reported_prop': 1,  #omega
         'relative_report': 0.014,  #delta
         'local_known_onsets': 0.16,  #p_w
         'travel_restriction': 63  #day on which travel restrictions have been put in place
        }
theta['travel_prop'] = theta['passengers'] / theta['pop_Wuhan']

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
        init_var['beta'] = Dirac(loc=self.initial_values['beta'])
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
        expected_obs['onset'] = Dirac(loc=x['D'] * self.omega * self.delta * self.pw)
        expected_obs['reported'] = Dirac(loc=x['C'] * self.omega * self.delta)

        obs_dist = StructDist(expected_obs)

        return obs_dist



if __name__ == '__main__':

    time_range = 82

    model = TransmissionModel(a=theta['brownian_vol'],
                              N=theta['pop_Wuhan'],
                              sigma=theta['incubation'],
                              kappa=theta['report'],
                              gamma=theta['recover'],
                              f=theta['travel_prop'],
                              omega=theta['reported_prop'],
                              delta=theta['relative_report'],
                              pw=theta['local_known_onsets'],
                              travel_restriction=theta['travel_restriction'],
                              initial_values=initial_values
                              )
    hidden_states, observations = model.simulate(time_range)

    end = True