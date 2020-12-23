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
                  'exp': 0,
                  'inf': 1,
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


# Define model equations

def sus(sus_prev, beta, inf, N):
    return max(sus_prev - beta * sus_prev * inf / N, 0)

def exp(t, exp_prev, beta, sus, inf, f, sigma, N, travel_restriction):
    return exp_prev + (1 - (t < travel_restriction) * f) * beta * sus * inf / N - 2 * sigma * exp_prev

def inf(inf_prev, exp_prev, sigma, gamma):
    return inf_prev + 2 * sigma * exp_prev - 2 * gamma * inf_prev

def Q(Q_prev, exp_prev, sigma, gamma, kappa):
    factor = np.exp(- gamma * kappa)
    return Q_prev + 2 * sigma * exp_prev * factor - kappa * Q_prev

def D(D_prev, exp_prev, sigma, gamma, kappa):
    factor = np.exp(- gamma * kappa)
    return D_prev + 2 * sigma * exp_prev * factor

def C(C_prev, Q_prev, kappa):
    return C_prev + kappa * Q_prev


# Define model object

class TransmissionModel(ssm.StateSpaceModel):

    def PX0(self):  # Distribution of X_0

        init_var = OrderedDict()
        init_var['beta'] = Dirac(loc=self.initial_values['beta'])
        init_var['sus'] = Dirac(loc=self.initial_values['sus'])
        init_var['exp'] = Dirac(loc=self.initial_values['exp'])
        init_var['inf'] = Dirac(loc=self.initial_values['inf'])
        init_var['Q'] = Dirac(loc=self.initial_values['Q'])
        init_var['D'] = Dirac(loc=self.initial_values['D'])
        init_var['C'] = Dirac(loc=self.initial_values['C'])

        init_dist = StructDist(init_var)

        return init_dist

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)

        lat_var = OrderedDict()
        lat_var['beta'] = LinearD(ExpD(Normal(scale=self.a)), a=xp['beta'])
        lat_var['inf'] = Cond(lambda x: Dirac(loc=inf(xp['inf'], xp['exp'], self.sigma, self.gamma)))
        lat_var['sus'] = Cond(lambda x: Dirac(loc=sus(xp['sus'], x['beta'], x['inf'], self.N)))
        lat_var['exp'] = Cond(
            lambda x: Dirac(loc=exp(t, xp['exp'], x['beta'], x['sus'], x['inf'], self.f, self.sigma, self.N,
                                    self.travel_restriction))
        )
        lat_var['Q'] = Cond(lambda x: Dirac(loc=Q(xp['Q'], xp['exp'], self.sigma, self.gamma, self.kappa)))
        lat_var['D'] = Cond(lambda x: Dirac(loc=D(xp['D'], xp['exp'], self.sigma, self.gamma, self.kappa)))
        lat_var['C'] = Cond(lambda x: Dirac(loc=C(xp['C'], xp['Q'], self.kappa)))

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