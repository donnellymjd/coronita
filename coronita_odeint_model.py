#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os, time, stat, io, glob, pickle
from scipy.stats import gamma, norm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy

from covid_data_helper import *
from coronita_model_helper import *


def noncohort_ode_model(initial_conditions, t_, r_ts, params):
    S, E, E_new, I_Mild, I_Sev, I_Fatal, H_Sev, H_Fatal, H_Admits, R, D = initial_conditions
    N = S + E + I_Mild + I_Sev + I_Fatal + H_Sev + H_Fatal + R + D
    I = I_Mild + I_Sev + I_Fatal
    H = H_Sev + H_Fatal

    try:
        r_t = r_ts.loc[int(t_)]
    except:
        print(int(t_))

    # d_til_death_t = (params['d_til_death'].value - params['d_til_death_curr'].value) \
    #                 / (1 + np.exp(params['mort_logistic_steepness'] * (t_ - 182))) \
    #                 + params['d_til_death_curr'].value

    ## FLOWS ##

    _sigma = 1 / params['d_incub'].value
    _gamma = 1 / params['d_infect'].value
    _beta = r_t * _gamma
    _nu = 1 / params['d_to_hosp'].value
    _rho = 1 / params['d_in_hosp'].value
    # _mu = 1 / d_til_death_t
    _mu = 1 / params['d_til_death'].value

    # p_fatal = (params['mort_rt'].value - params['mort_rt_curr'].value) \
    #           / (1 + np.exp(params['mort_logistic_steepness'] * (t_ - 182))) \
    #           + params['mort_rt_curr'].value
    p_fatal = params['mort_rt'].value
    if p_fatal <= 0:
        print(t_, p_fatal)
    p_recov_sev = params['hosp_rt'].value - params['mort_rt'].value
    p_recov_mild = 1 - p_fatal - p_recov_sev

    dSdt = -min(_beta * I, S)
    dEdt = min(_beta * I, S) - _sigma * E
    E_new = min(_beta * I, S) - E_new

    dI_Milddt = p_recov_mild * _sigma * E - _gamma * I_Mild
    dI_Sevdt = p_recov_sev * _sigma * E - _nu * I_Sev
    dI_Fataldt = p_fatal * _sigma * E - _nu * I_Fatal

    dH_Sevdt = _nu * I_Sev - _rho * H_Sev
    dH_Fataldt = _nu * I_Fatal - _mu * H_Fatal
    H_Admits = _nu * I_Sev + _nu * I_Fatal - H_Admits

    dRdt = _gamma * I_Mild + _rho * H_Sev
    dDdt = _mu * H_Fatal
    return [dSdt, dEdt, E_new, dI_Milddt, dI_Sevdt, dI_Fataldt, dH_Sevdt, dH_Fataldt, H_Admits, dRdt, dDdt]


def run_model(params, initial_conditions, time_dict, r_ts):
    start_dt_fmt = pd.Timestamp('2020-01-01') + pd.Timedelta(days=params['start_dt'].value)
    #     print(start_dt_fmt)

    r_ts_nodt = r_ts.copy()
    r_ts_nodt.index = (r_ts_nodt.index - start_dt_fmt).days

    sol = odeint(noncohort_ode_model, initial_conditions, time_dict['tspan'], args=(r_ts_nodt, params,))
    return sol

def sol_array_to_df(sol, params, time_dict):
    df_sol = pd.DataFrame(sol,
                          columns=['N', 'E', 'E_new',
                                   'I_Mild', 'I_Sev', 'I_Fatal',
                                   'H_Sev', 'H_Fatal', 'H_Admits', 'R', 'D'],
                          index=pd.date_range(
                              (pd.Timestamp('2020-01-01') + pd.Timedelta(days=params['start_dt'].value)),
                              periods=time_dict['days'] / time_dict['granularity'],
                              freq=f"{time_dict['hrs_per_point']}H")
                          )
    agg_dict = {
        'E': 'mean',
        'E_new': 'sum',
        'I_Mild': 'mean',
        'I_Sev': 'mean',
        'I_Fatal': 'mean',
        'H_Sev': 'mean',
        'H_Fatal': 'mean',
        'H_Admits': 'sum',
        'R': 'max',
        'D': 'max'}
    df_sol_daily = df_sol.resample('D').agg(agg_dict)
    df_sol_daily['deaths_tot'] = df_sol_daily['D']
    df_sol_daily['deaths_daily'] = df_sol_daily['D'].diff()
    df_sol_daily['hosp_admits'] = df_sol_daily['H_Admits']
    df_sol_daily['hosp_concur'] = df_sol_daily[['H_Sev', 'H_Fatal']].sum(axis=1)
    return df_sol_daily


def report_model(r_ts, df_sol, model_dict, params, time_dict):
    r_ts.plot();
    plt.show()

    for series in ['hosp_concur', 'hosp_admits', 'deaths_tot', 'deaths_daily']:
        df_sol[series].plot(title=series, label='Model', legend=True)
        try:
            model_dict['df_hist'][series].plot(title=series, label='Reported Data', legend=True)
        except:
            pass
        plt.show()

    # mort_rt_t = ((params['mort_rt'].value - params['mort_rt_curr'].value)
    #              / (1 + np.exp(params['mort_logistic_steepness'] * (time_dict['tspan'] - 182)))
    #              + params['mort_rt_curr'].value)
    mort_rt_t = params['mort_rt'].value
    mort_rt_idx = pd.date_range(
                              (pd.Timestamp('2020-01-01') + pd.Timedelta(days=params['start_dt'].value)),
                              periods=time_dict['days'] / time_dict['granularity'],
                              freq=f"{time_dict['hrs_per_point']}H")
    mort_rt_t = pd.Series(mort_rt_t, index=mort_rt_idx)
    mort_rt_t.plot(title='mort_rt_t')
    plt.show()
    #
    # d_til_death_t = ((params['d_til_death'].value - params['d_til_death_curr'].value)
    #                  / (1 + np.exp(params['mort_logistic_steepness'] * (time_dict['tspan'] - 182)))
    #                  + params['d_til_death_curr'].value)
    d_til_death_t = params['d_til_death'].value
    d_til_death_idx = pd.date_range(
                              (pd.Timestamp('2020-01-01') + pd.Timedelta(days=params['start_dt'].value)),
                              periods=time_dict['days'] / time_dict['granularity'],
                              freq=f"{time_dict['hrs_per_point']}H")
    d_til_death_t = pd.Series(d_til_death_t, index=d_til_death_idx)
    d_til_death_t.plot(title='d_til_death_t')
    plt.show()

    print(df_sol.iloc[-1])
    return


def error(params, initial_conditions, time_dict, r_ts, data):
    data_dropna = data.copy().replace(0, np.nan).dropna(axis=1, thresh=15).dropna(how='any')
    # data_dropna = data_dropna.clip(lower=data_dropna.quantile(0.01), upper=data_dropna.quantile(0.99), axis=1)
    #     data = data.replace(0,np.nan)
    sol = run_model(params, initial_conditions, time_dict, r_ts)
    df_sol = sol_array_to_df(sol, params, time_dict)

    error = (df_sol - data_dropna).dropna(axis=1, how='all')
    error = error.div(data_dropna.max())

    ### FAILED NORMALIZATION METHODS ###

    ## LOG DIFFERENCES METHOD ##
    # Note: Looks like this isn't working bc small gross errors appear as large log diff errors #
    # error = (df_sol.apply(np.log) - data_dropna.apply(np.log))
    ############################

    #     error_normalized = error.div(data_dropna).mul(1e3)
    #     error_normalized = error.div(error.mean())
    #     error_normalized = np.log(error).mul(1e3)

    ####################################

    ### REPORT WITHIN ERROR FUNCTION ###
    #     display(result.params)

    #     for col in data.columns:
    #         ax = data[col].plot(title=col)
    #         df_sol[col].plot(ax=ax); plt.show()
    #     error.plot(); plt.show()
    #     print(f'Chisq: {np.nansum(np.square(error_normalized)): 0.0f}')
    ####################################
    return error.to_numpy()
