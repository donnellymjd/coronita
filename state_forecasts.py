#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os, time, stat, io, glob, pickle
from scipy.stats import gamma, norm

from covid_data_helper import *
from coronita_model_helper import *


## DATA INGESTION ##

df_st_testing = get_covid19_tracking_data()

df_census = get_census_pop()

df_counties = get_complete_county_data()

counties_geo = get_counties_geo()

df_jhu_counties = get_jhu_counties()

df_st_testing_fmt = df_st_testing.copy()
df_st_testing_fmt = df_st_testing_fmt.rename(columns={'death':'deaths','positive':'cases'}).unstack('code')

try:
    df_interventions = get_state_policy_events()
except:
    df_interventions = pd.DataFrame()

df_goog_mob_us = get_goog_mvmt_us()
df_goog_mob_state = get_goog_mvmt_state(df_goog_mob_us)
df_goog_mob_us = df_goog_mob_us[df_goog_mob_us.state.isnull()].set_index('dt')

df_hhs_hosp = get_hhs_hosp()

#######################

## MODEL PARAMETERS ##

covid_params = {}
covid_params['d_incub'] = 3.
covid_params['d_infect'] = 4.
covid_params['mort_rt'] = 0.01
covid_params['d_in_hosp'] = 11
covid_params['hosp_rt'] = 0.04
covid_params['d_to_hosp'] = 7.0
covid_params['d_in_hosp_mild'] = 11.0
covid_params['icu_rt'] = 13./41.
covid_params['d_in_icu'] = 13.0
covid_params['vent_rt'] = 0.4
covid_params['d_til_death'] =  30.0 #17.0
covid_params['policy_trigger'] = True
covid_params['policy_trigger_once'] = True
days_to_forecast = 150

#######################

### RUN MODEL ###
df_fore_allstates = pd.DataFrame()

try:
    list_of_files = glob.glob('./output/df_fore_allstates_*.pkl') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print('last forecast: ', latest_file)
    df_prevfore_allstates = pd.read_pickle(latest_file)
except:
    if 'df_fore_allstates' in globals().keys():
        if df_fore_allstates.shape[0] > 0:
            df_prevfore_allstates = df_fore_allstates.copy()

allstate_model_dicts = {}
df_rts_allregs = pd.DataFrame()
df_wavg_rt_conf_allregs = pd.DataFrame()

for state in df_census.state.unique():
    print(state)
    
    model_dict = make_model_dict_state(state, abbrev_us_state, df_census, df_st_testing_fmt, df_hhs_hosp,
                                       covid_params, days_to_forecast,
                                       df_mvmt=df_goog_mob_state
                                     , df_interventions=df_interventions
                                      )

    this_reg_df_rts = pd.DataFrame(model_dict['df_rts'].stack(), columns=[state])
    this_reg_df_wavg = pd.DataFrame(
        model_dict['df_rts_conf'].sort_index().unstack('metric')['weighted_average'].stack(), columns=[state])

    df_rts_allregs = pd.concat([df_rts_allregs, this_reg_df_rts], axis=1)
    df_wavg_rt_conf_allregs = pd.concat([df_wavg_rt_conf_allregs, this_reg_df_wavg], axis=1)

    try:
        first_guess = df_prevfore_allstates[state].first_valid_index()[0]
    except:
        # first_guess = pd.Timestamp('2020-02-17')
        local_r0_date = model_dict['df_rts'].loc['2020-02-01':'2020-04-30', 'weighted_average'].idxmax()
        first_guess = local_r0_date - pd.Timedelta(days=28)

    model_dict = model_find_start(first_guess, model_dict)
    df_agg = model_dict['df_agg']
    df_all_cohorts = model_dict['df_all_cohorts']

    print('Peak Hospitalization Date: ', df_agg.hospitalized.idxmax().strftime("%d %b, %Y"))
    print('Peak Hospitalization #: {:.0f}'.format(df_agg.hospitalized.max()))
    print('Peak ICU #: {:.0f}'.format(df_agg.icu.max()))
    print('Peak Ventilator #: {:.0f}'.format(df_agg.vent.max()))

    model_dict['chart_title'] = r'No Change in Future $R_{t}$ Until Reaching Hospital Capacity Triggers Lockdown'

    allstate_model_dicts[state] = model_dict
    df_fore_allstates = pd.concat([df_fore_allstates,pd.DataFrame(df_agg.stack(), columns=[state])], axis=1)

#######################

### Add US Country Level Entries Before Saving ###
df_fore_us = df_fore_allstates.sum(axis=1, skipna=True).unstack('metric').dropna(how='all')
tot_pop = df_fore_us[['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered']].sum(axis=1)
max_tot_pop = tot_pop.max()
df_fore_us.loc[tot_pop<max_tot_pop, 'susceptible'] = df_fore_us['susceptible'] + (max_tot_pop - tot_pop)

last_fore_dt = df_fore_allstates.sum(axis=1, skipna=False).unstack('metric').last_valid_index()
df_fore_us = df_fore_us.loc[:last_fore_dt]

df_fore_us_stack = pd.DataFrame(df_fore_us.stack(), columns=['US'])
df_fore_allstates = pd.concat([df_fore_allstates, df_fore_us_stack], axis=1)

model_dict = make_model_dict_us(df_census, df_st_testing_fmt, df_hhs_hosp, covid_params, d_to_forecast=75,
                               df_mvmt=df_goog_mob_us, df_interventions=df_interventions)
model_dict['df_agg'] = df_fore_us
model_dict['chart_title'] = r'No Change in Future $R_{t}$ Until Reaching Hospital Capacity Triggers Lockdown'
allstate_model_dicts['US'] = model_dict

this_reg_df_wavg = pd.DataFrame(
    model_dict['df_rts_conf'].sort_index().unstack('metric')['weighted_average'].stack(), columns=['US'])
df_wavg_rt_conf_allregs = pd.concat([df_wavg_rt_conf_allregs, this_reg_df_wavg], axis=1)
###################################################

### Save Output ###
df_rts_allregs.index.names = ['dt','metric']

df_wavg_rt_conf_allregs.unstack('metric').to_csv(
    './output/df_wavg_rt_conf_allregs_{}.csv'.format(pd.Timestamp.today().strftime("%Y%m%d")),
    encoding='utf-8')
df_wavg_rt_conf_allregs.to_pickle('./output/df_wavg_rt_conf_allregs_{}.pkl'.format(
    pd.Timestamp.today().strftime("%Y%m%d")))

df_fore_allstates.unstack('metric').to_csv(
    './output/df_fore_allstates_{}.csv'.format(pd.Timestamp.today().strftime("%Y%m%d")),
    encoding='utf-8')
df_fore_allstates.unstack('metric').to_csv(
    '../COVIDoutlook/download/df_fore_allstates_{}.csv'.format(pd.Timestamp.today().strftime("%Y%m%d")),
    encoding='utf-8')
df_fore_allstates.to_pickle('./output/df_fore_allstates_{}.pkl'.format(pd.Timestamp.today().strftime("%Y%m%d")))

asmd_filename = './output/allstate_model_dicts_{}.pkl'.format(pd.Timestamp.today().strftime("%Y%m%d"))

if df_interventions.shape[0] > 0:
    df_interventions.to_csv('../COVIDoutlook/download/df_interventions.csv', encoding='utf-8')
else:
    print('!!!!!Could not update df_interventions!!!!')

with open(asmd_filename, 'wb') as handle:
    pickle.dump(allstate_model_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

os.system('say -v "Victoria" "Your forecasts are ready."')

######################

os.system('python web_gen_covidoutlook.py &') # 'python web_gen_covidoutlook.py 2>&1 | tee -a web_gen_covidoutlook.log &'
os.system('python web_gen_personal.py &')