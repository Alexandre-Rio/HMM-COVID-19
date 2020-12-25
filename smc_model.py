'''
Adapted from R scripts on https://github.com/adamkucharski/2020-ncov
'''

import numpy as np
import pandas as pd
from scipy.stats import poisson, binom, nbinom
import matplotlib.pyplot as plt

## Load data
from data import *

## Variables

t_period = (date_range[-1] - date_range[0]).days + 1
wuhan_travel_time = (pd.Timestamp('2020-01-23')-date_range[0]).days+1
top_risk = travel_data

theta = {'betavol': 0.395,  # Brownian volatility
        'r0': 2.5,  # Initial R0
        'gentime': 7,
        'init_cases': 1,
        'recover': 1 / 2.9,  #gamma
        'report': 1 / 6.1,  # kappa
        'incubation': 1 / 5.2,  # sigma
        'outbound_travel': 3300,  # Daily outbound passengers
        'population_travel': 1.10e7,  # Population of Wuhan (N)
        'travel_frac': None,  # population_travel / outbound_travel
        'local_rep_prop': 0.0066,
        'onset_prop': 0.16,  #p_w
        'onset_prop_int': 0.47,  #p_T
        'confirmed_prop': 1,  # Eventually, every case is confirmed
        'report_local': 1 / 6.1,  # kappa local
        'r0_decline': 1,
        'rep_local_var': 1,
        'pre_symp': 0,
        'beta': None}

theta['passengers'] = theta['outbound_travel']
theta['pop_travel'] = theta['population_travel']
theta["travel_frac"] = theta["passengers"] / theta["pop_travel"] # Estimate fraction that travel
theta["beta"] = theta["r0"]*theta["recover"]  # Scale initial value of R0

lat_var_names = ["sus", "tr_exp1", "tr_exp2", "exp1", "exp2", "inf1", "inf2", "tr_waiting", "cases", "reports", "waiting_local", "cases_local", "reports_local"] # also defines groups to use in model
lat_var_id = {param : lat_var_names.index(param) for param in lat_var_names}

data_dict = {'local_case_data_onset' : local_case_data_onset,
            'local_case_data_conf' : local_case_data_conf,
            'int_case_onset' : int_case_onset,
            'int_case_conf' : int_case_conf,
            'int_case_onset_scale' : int_case_onset_scale,
            'local_case_data_onset_scale' : local_case_data_onset_scale,
            'flight_info' : flight_info,
            'flight_prop' : flight_prop}


## Process model for simulation

def process_model(t_start,t_end,dt,theta,simTabi,simzetaA,travelF) :

    simTab = simTabi.copy()

    # From table to vectors
    susceptible_t = simTab[:,lat_var_id['sus']] # input function
    exposed_t1    = simTab[:,lat_var_id['exp1']] # input function
    exposed_t2    = simTab[:,lat_var_id['exp2']] # input function
    tr_exposed_t1 = simTab[:,lat_var_id['tr_exp1']] # input function
    tr_exposed_t2 = simTab[:,lat_var_id['tr_exp2']] # input function

    infectious_t1 = simTab[:,lat_var_id['inf1']] # input function
    infectious_t2 = simTab[:,lat_var_id['inf2']] # input function

    tr_waiting_t = simTab[:,lat_var_id['tr_waiting']] # input function
    cases_t = simTab[:,lat_var_id['cases']] # input function
    reports_t = simTab[:,lat_var_id['reports']] # input function

    waiting_local_t = simTab[:,lat_var_id['waiting_local']] # input function
    cases_local_t = simTab[:,lat_var_id['cases_local']] # input function
    reports_local_t = simTab[:,lat_var_id['reports_local']] # input function

    # scale transitions
    inf_rate = (simzetaA/theta["pop_travel"])*dt
    inc_rate = theta["incubation"]*2*dt
    rec_rate = theta["recover"]*2*dt
    rep_rate_local = theta["report_local"]*dt
    rep_rate = theta["report"]*dt
    prob_rep = np.exp(-theta["report"]*theta["recover"]) # probability case is reported rather than recovers
    prob_rep_local = np.exp(-theta["report_local"]*theta["recover"]) # probability case is reported rather than recovers

    for _ in np.arange(t_start+dt, t_end+dt, dt) :

        # transitions
        S_to_E1 = susceptible_t*(theta["pre_symp"]*tr_exposed_t2+infectious_t1+infectious_t2)*inf_rate # stochastic transmission

        # Delay until symptoms
        E1_to_E2 = exposed_t1*inc_rate # as two compartments
        E2_to_I1 = exposed_t2*inc_rate

        E1_to_E2_tr = tr_exposed_t1*inc_rate # as two compartments
        E2_to_I1_tr = tr_exposed_t2*inc_rate

        # Delay until recovery
        I1_to_I2 = infectious_t1*rec_rate
        I2_to_R = infectious_t2*rec_rate

        # Delay until reported
        W_to_Rep = tr_waiting_t*rep_rate

        W_to_Rep_local = waiting_local_t*rep_rate_local

        # Process model for SEIR
        susceptible_t = susceptible_t - S_to_E1
        exposed_t1 = exposed_t1 + S_to_E1*(1-travelF) - E1_to_E2
        exposed_t2 = exposed_t2 + E1_to_E2 - E2_to_I1
        tr_exposed_t1 = tr_exposed_t1 + S_to_E1*travelF - E1_to_E2_tr
        tr_exposed_t2 = tr_exposed_t2 + E1_to_E2_tr - E2_to_I1_tr

        infectious_t1 = infectious_t1 + E2_to_I1 - I1_to_I2
        infectious_t2 = infectious_t2 + I1_to_I2 - I2_to_R

        # Case tracking - including removal of cases within Q compartment
        waiting_local_t = waiting_local_t + E2_to_I1*prob_rep_local - W_to_Rep_local
        cases_local_t = cases_local_t + E2_to_I1*prob_rep_local
        reports_local_t = reports_local_t + W_to_Rep_local

        tr_waiting_t = tr_waiting_t + E2_to_I1_tr*prob_rep - W_to_Rep
        cases_t = cases_t + E2_to_I1_tr*prob_rep
        reports_t = reports_t + W_to_Rep

    simTab[:,lat_var_id["sus"]] = susceptible_t # output
    simTab[:,lat_var_id["exp1"]] = exposed_t1 # output
    simTab[:,lat_var_id["exp2"]] = exposed_t2 # output
    simTab[:,lat_var_id["tr_exp1"]] = tr_exposed_t1 # output
    simTab[:,lat_var_id["tr_exp2"]] = tr_exposed_t2 # output
    simTab[:,lat_var_id["inf1"]] = infectious_t1 # output
    simTab[:,lat_var_id["inf2"]] = infectious_t2 # output

    simTab[:,lat_var_id["tr_waiting"]] = tr_waiting_t # output
    simTab[:,lat_var_id["cases"]] = cases_t # output
    simTab[:,lat_var_id["waiting_local"]] = waiting_local_t # output
    simTab[:,lat_var_id["reports"]] = reports_t # output
    simTab[:,lat_var_id["cases_local"]] = cases_local_t # output
    simTab[:,lat_var_id["reports_local"]] = reports_local_t # output

    return simTab


## SMC model

def smc_model(theta, nn, dt = 1) :

    # Assumptions - using daily growth rate
    ttotal = t_period

    storeL = np.zeros((nn, ttotal, len(lat_var_names)))

    # Add initial condition
    storeL[:,0,lat_var_id['inf1']] = theta["init_cases"]/2
    storeL[:,0,lat_var_id['inf2']] = theta["init_cases"]/2
    storeL[:,0,lat_var_id['sus']] = (theta["pop_travel"] - theta["init_cases"])

    simzeta = np.random.normal(0, theta['betavol'], size = [ttotal, nn])
    simzeta[0,:] = np.exp(simzeta[0,:])*theta["beta"] # define Initial Condition

    # Initialize latent variables
    S_traj = np.full((ttotal,1), None)
    C_local_traj = np.full((ttotal,1), None)
    Rep_local_traj = np.full((ttotal,1), None)
    C_traj = np.full((ttotal,1), None)
    Rep_traj = np.full((ttotal,1), None)
    E_traj = np.full((ttotal,1), None)
    I_traj = np.full((ttotal,1), None)
    beta_traj = np.full((ttotal,1), None)
    w = np.full((nn,ttotal), None)
    w[:,0] = 1
    W = np.full((nn,ttotal), None)
    A = np.full((nn,ttotal), None) # particle parent matrix
    l_sample = np.full((ttotal), None)
    lik_values = np.full((ttotal), None)

    # Iterate through steps
    for tt in range(1, ttotal) :

        # Add random walk on transmission ?
        simzeta[tt,:] = simzeta[tt-1,:]*np.exp(simzeta[tt,:])  # Compute R_0(t) from R_0(t-1)

        # travel restrictions in place?
        travelF = theta["travel_frac"] if tt < wuhan_travel_time else 0

        # run process model
        storeL[:,tt,:] = process_model(tt-1, tt, dt, theta, storeL[:,tt-1,:], simzeta[tt,:], travelF)

        # calculate weights
        w[:,tt] = AssignWeights(data_dict, storeL, nn, theta, tt)

        # check likelihood isn't NA
        wmax = w[:,tt].max()
        if pd.isnull(wmax) or wmax == 0 :
            print('SMC stopped')
            return None

        # normalise particle weights
        W[:,tt] = w[:,tt]/w[:,tt].sum()

        # resample particles by sampling parent particles according to weights:
        A[:, tt] = np.random.choice(range(nn), p=W[:,tt].astype(float), replace=True)

        # Resample particles for corresponding variables
        storeL[:, tt, :] = storeL[A[:, tt].astype(int),tt,:]
        simzeta[tt, :] = simzeta[tt, A[:, tt].astype(int)] #- needed for random walk on beta

    # Estimate likelihood:
    for tt in range(ttotal) :
        lik_values[tt] = np.log(w[:,tt].sum())  # log-likelihoods

    likelihood0 = -ttotal*np.log(nn) + np.sum(lik_values)  # add full averaged log-likelihoods

    # Sample latent variables:
    ttotal -= 1
    locs = np.random.choice(range(nn), p = W[:,ttotal].astype(float), replace=True)
    l_sample[ttotal] = locs
    S_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['sus']]
    C_local_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['cases_local']]
    Rep_local_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['reports_local']]
    C_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['cases']]
    Rep_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['reports']]
    E_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['exp2']] + storeL[l_sample[ttotal],ttotal,lat_var_id['exp1']]
    I_traj[ttotal,:] = storeL[l_sample[ttotal],ttotal,lat_var_id['inf1']] + storeL[l_sample[ttotal],ttotal,lat_var_id['inf2']]
    beta_traj[ttotal,:] = simzeta[ttotal,l_sample[ttotal]]

    for ii in range(ttotal, 0, -1):
        l_sample[ii-1] = A[l_sample[ii],ii] # have updated indexing
        S_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['sus']]
        C_local_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['cases_local']]
        Rep_local_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['reports_local']]
        C_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['cases']]
        Rep_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['reports']]
        E_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['exp2']] + storeL[l_sample[ii-1],ii-1,lat_var_id['exp1']]
        I_traj[ii-1,:] = storeL[l_sample[ii-1],ii-1,lat_var_id['inf1']] + storeL[l_sample[ii-1],ii-1,lat_var_id['inf2']]
        beta_traj[ii-1,:] = simzeta[ii-1,l_sample[ii-1]]

    return [S_traj, C_local_traj, Rep_local_traj, C_traj, Rep_traj, I_traj, beta_traj, likelihood0]


## Assign weights

def pmax(X, a) :
    return np.array([x if x > a else a for x in X])

def AssignWeights(data_dict,storeL,nn,theta,tt) :
    '''
    Compute the likelihood of the model as described in the appendix. For the variables described below:
    Expected values of the are computed from the model and compared to the observed values.
    Variables on which the model is fitted:
    - Confirmed local cases (by onset)
    - Confirmed local cases (by confirmation)
    - International onsets (total)
    - International confirmed cases (by country)
    - Probability of infection from evacuation flights
    :param data_dict: Dictionary of observed data
    :param storeL: Model values
    :param nn: Number of particles
    :param theta: Parameters
    :param tt: Current time step
    :return: Likelihood of the model
    '''
    # Gather data (true/observed data)
    local_case_data_tt = data_dict['local_case_data_onset'].values[tt]
    local_case_data_conf_tt = data_dict['local_case_data_conf'].values[tt]
    case_data_tt = data_dict['int_case_onset'].values[tt]
    rep_data_tt = data_dict['int_case_conf'].values[tt]

    flight_info_tt = data_dict['flight_info'].values[tt]
    flight_prop_tt = data_dict['flight_prop'].values[tt,:]

    # Scale for reporting lag
    case_data_tt_scale = 1 #data_list$int_case_onset_scale[tt] # deprecated
    local_case_data_tt_scale = 1 #data_list$local_case_data_onset_scale[tt] # deprecated

    # Gather variables (values from model)
    case_localDiff = storeL[:,tt,lat_var_id["cases_local"]] - storeL[:,tt-1,lat_var_id["cases_local"]]
    rep_local = storeL[:,tt,lat_var_id["reports_local"]]
    repDiff_local = storeL[:,tt,lat_var_id["reports_local"]] - storeL[:,tt-1,lat_var_id["reports_local"]]
    caseDiff = storeL[:,tt,lat_var_id["cases"]] - storeL[:,tt-1,lat_var_id["cases"]]
    repDiff = storeL[:,tt,lat_var_id["reports"]] - storeL[:,tt-1,lat_var_id["reports"]]

    # Prevalence - scale by asymptomatics - second half only // storeL[:,tt,"exp1"] +
    inf_prev = storeL[:,tt,lat_var_id["exp1"]] + storeL[:,tt,lat_var_id["exp2"]] + (storeL[:,tt,lat_var_id["inf1"]] + storeL[:,tt,lat_var_id["inf2"]])*(1-theta["confirmed_prop"])

    # Check for positivity
    c_local_val = pmax(case_localDiff, 0)
    c_val = pmax(caseDiff, 0)
    rep_val = pmax(repDiff, 0)
    rep_val_local = pmax(repDiff_local, 0)
    r_local_rep_cum = rep_local

    # Local confirmed cases (by onset)

    if not pd.isnull(local_case_data_tt) :
        expected_val = c_local_val * theta["confirmed_prop"] * theta["onset_prop"] * theta["local_rep_prop"] * local_case_data_tt_scale # scale by reporting proportion and known onsets
        loglikSum_local_onset = np.log(poisson.pmf(local_case_data_tt, mu = expected_val))
    else :
        loglikSum_local_onset = 0

    # Local confirmed cases (by confirmation) -- HOLD OUT FOR NOW AS LIKELIHOOD LOW

    if not pd.isnull(local_case_data_conf_tt) :
        expected_val = rep_val_local * theta["confirmed_prop"] * theta["local_rep_prop"] # scale by reporting proportion and known onsets
        mu, size = expected_val, 1/theta["rep_local_var"]
        loglikSum_local_conf = np.log(nbinom.pmf(local_case_data_conf_tt, n = size, p = size/(mu+size)))
    else :
        loglikSum_local_conf = 0

    # International confirmed cases (by country)

    # Do location by location
    x_scaled = theta["confirmed_prop"]*rep_val
    x_expected = np.array([x * top_risk.values.reshape(len(top_risk)) for x in x_scaled]) # expected exported cases in each location

    # # Note here rows are particles, cols are locations.
    x_lam = x_expected.flatten() # flatten data on expectation
    y_lam = np.array([rep_data_tt for _ in range(nn)]).flatten()
    # np.full(nn, rep_data_tt) #dim(y_lam) = NULL

    # Calculate likelihood
    loglik = np.log(poisson.pmf(y_lam, mu = x_lam))
    loglikSum_int_conf = loglik.reshape((nn,-1)).sum(0)

    # International onsets (total)
    if not pd.isnull(case_data_tt) :
        x_scaled = theta["confirmed_prop"] * c_val * theta["onset_prop_int"]
        x_expected = np.array([x * top_risk.values.reshape(len(top_risk)) for x in x_scaled]).T
        x_lam = x_expected.sum(0)
        y_lam = case_data_tt

        # Calculate likelihood
        loglik = np.log(poisson.pmf(y_lam, mu = x_lam))
        loglikSum_inf_onset = loglik.repeat(nn/len(loglik)).reshape((nn,-1)).sum(1) #loglik.reshape((nn,-1)).sum(0)
    else :
        loglikSum_inf_onset = 0

    # Additional probablity infections from evacuation flights

    if not pd.isnull(flight_info_tt) :
        prob_inf = pmax(inf_prev/theta["pop_travel"],0) # ensure >=0
        loglikSum_flight_info = np.log(binom.pmf(flight_prop_tt[0],flight_prop_tt[1],p=prob_inf))
        loglikSum_flight_info = np.array([-np.inf if pd.isnull(l) else l for l in loglikSum_flight_info]) # Set NA = -Inf
    else :
        loglikSum_flight_info = 0

    # Tally up likelihoods
    # loglikSum_local_conf
    loglikSum = loglikSum_local_onset + loglikSum_inf_onset + loglikSum_flight_info #+ loglikSum_local_conf #+ loglikSum_int_conf #+ loglikSum_local_conf

    return np.exp(loglikSum) # convert to normal probability


## Workspace

if __name__ == '__main__':
    r0_mean = []
    t_step = 1
    for i in range(10):
        print(i)
        output_smc = smc_model(theta,
                               nn=1000,  # number of particles
                               dt=t_step)
        r0_mean += [output_smc[-2] / theta["recover"]]
        plt.plot(date_range, r0_mean[-1], 'b')
    plt.plot(date_range, np.mean(r0_mean, 0), 'r')
    plt.show()










