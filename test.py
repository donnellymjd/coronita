import pandas as pd
import numpy as np
import os, glob, pickle

list_of_files = glob.glob('./output/allstate_model_dicts_*.pkl')  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
with open(latest_file, 'rb') as handle:
    allstate_model_dicts = pickle.load(handle)

start_dt = '2020-02-04'
model_dict = allstate_model_dicts['NJ']
exposed_0 = 1
infectious_0 = 0

# def seir_model_cohort(start_dt, model_dict, exposed_0=100, infectious_0=100):
next_hospitalized = 0

vax_new_people_daily = model_dict['df_hist']['vax_initiated'].diff() \
    .reindex(model_dict['df_vax_fore'].index).fillna(model_dict['df_vax_fore']['trend'].diff())
vax_new_people_daily = vax_new_people_daily.reindex(index=pd.date_range(start_dt, vax_new_people_daily.index[-1]))
vax_new_people_daily = vax_new_people_daily.fillna(method='bfill')
next_recovered_vaxxed = 0
next_susceptible_vaxxed = vax_new_people_daily.loc[start_dt] if start_dt in vax_new_people_daily.index else 0


recovered_vaxxed = [next_recovered_vaxxed]
susceptible_vaxxed = [next_susceptible_vaxxed]

N = model_dict['tot_pop']
E, E_new = [[exposed_0], [exposed_0]]
I_Mild, I_Sev, I_Fatal, H_Sev, H_Fatal, H_Admits, R, D = [[0]] * 8
S = [N[-1] - E[-1]]

t = np.linspace(0, model_dict['d_to_forecast'], model_dict['d_to_forecast'] + 1)

r_t = pd.Series(np.nan, index=pd.date_range(start_dt,
                                            max(start_dt + pd.Timedelta(days=model_dict['d_to_forecast']),
                                                pd.Timestamp('2021-07-04'))))

if 'rt_scenario' in model_dict['df_rts'].columns:
    r_t = r_t.fillna(model_dict['df_rts']['rt_scenario']).fillna(method='bfill').fillna(method='ffill')
else:
    local_r0_date = model_dict['df_rts'].loc['2020-02-01':'2020-04-30', 'weighted_average'].idxmax()
    # print('local_r0_date: ', local_r0_date)
    r_t = r_t.fillna(model_dict['df_rts'].loc[local_r0_date:, 'weighted_average'])
    r_t = r_t.fillna(method='bfill').fillna(method='ffill')
r_t_preimmune = r_t.copy()
# print(r_t.last_valid_index())

last_obs_rt = model_dict['df_rts']['weighted_average'].last_valid_index()

model_dict['df_rts'] = model_dict['df_rts'].reindex(r_t.index)
model_dict['df_rts']['policy_triggered'] = 0
model_dict['hosp_cap_dt'] = None

last_r = r_t.iloc[0]

for t_ in t[:-1]:
    cohort_strt = start_dt + pd.Timedelta(days=t_)
    # print(f'cohort_strt: {cohort_strt}, next_suspop: {next_suspop}')

    if (model_dict['covid_params']['policy_trigger']
            and (cohort_strt > last_obs_rt)):

        if 'hosp_beds_avail' in model_dict['df_hist'].columns:
            covid_hosp_capacity = \
            model_dict['df_hist']['hosp_beds_avail'].replace(0, np.nan).rolling(7).mean().dropna().iloc[-1]
            covid_hosp_capacity = covid_hosp_capacity + model_dict['df_hist']['hosp_concur'].dropna().iloc[-1]
        else:
            tot_hosp_capacity = model_dict['tot_pop'] / 1000 * 2.7
            covid_hosp_capacity = tot_hosp_capacity * 0.2

        if ((next_hospitalized > covid_hosp_capacity)
                or (model_dict['covid_params']['policy_trigger_once']
                    and model_dict['df_rts']['policy_triggered'].sum() > 1)):
            r_t.loc[cohort_strt] = 0.9
            model_dict['df_rts'].loc[cohort_strt, 'policy_triggered'] = 1
            if model_dict['hosp_cap_dt'] == None:
                model_dict['hosp_cap_dt'] = cohort_strt

    # ACCOUNT FOR EFFECT OF IMMUNITY IN FORECAST PERIOD #
    # Math:
    # rt / (suspop_lastdayofobs / tot_pop) * (suspop_t / tot_pop)
    # rt * (tot_pop / suspop_lastdayofobs) * (suspop_t / tot_pop)
    # # tot_pop cancels
    # rt * suspop_t / suspop_lastdayofobs
    if cohort_strt <= last_obs_rt:
        r_t_preimmune.loc[cohort_strt] = r_t.loc[cohort_strt] / (
                next_suspop / model_dict['tot_pop']) if next_suspop > 0 else 1.0

    if cohort_strt == last_obs_rt:
        suspop_lastdayofobs = next_suspop if next_suspop > 0 else 1.0
        # r_t_preimmune = r_t.div(suspop_lastdayofobs / model_dict['tot_pop'])
        r_t_preimmune.loc[cohort_strt + pd.Timedelta(days=1):] = np.nan

        current_r0 = model_dict['covid_params']['voc_transmissibility'] * model_dict['covid_params']['basic_r0']
        current_r0 = max(min(current_r0, 3.0), 2.0)
        model_dict['covid_params']['current_r0'] = current_r0
        # print(f'current_r0 {current_r0}')
        if r_t_preimmune.loc[r_t_preimmune.last_valid_index()] > current_r0:
            r_t_preimmune.loc['2021-07-04':] = r_t_preimmune.loc[r_t_preimmune.last_valid_index()]
        else:
            r_t_preimmune.loc['2021-07-04':] = current_r0
        r_t_preimmune = r_t_preimmune.interpolate()
        r_t_nochange = r_t.copy()
        # if '2021-07-04' in r_t_preimmune.index:
        #     print(r_t_preimmune.loc[cohort_strt], r_t_preimmune.loc['2021-07-04'])
        # print(f'{cohort_strt} next_suspop: {next_suspop}')
    elif cohort_strt > last_obs_rt:
        r_t_nochange.loc[cohort_strt] = r_t_nochange.loc[cohort_strt] * (next_suspop / suspop_lastdayofobs)
        r_t.loc[cohort_strt] = r_t_preimmune.loc[cohort_strt] * (next_suspop / model_dict['tot_pop'])

    this_r = r_t.loc[cohort_strt]

    if this_r != last_r:
        last_r = this_r

    params = model_dict['covid_params']

    ## RATES ##
    _sigma = 1 / params['d_incub']
    _gamma = 1 / params['d_infect']
    _beta = this_r * _gamma
    _nu = 1 / params['d_to_hosp']
    _rho = 1 / params['d_in_hosp']
    _mu = 1 / params['d_til_death']
    p_fatal = params['mort_rt']

    if p_fatal <= 0:
        print(t_, p_fatal)
    p_recov_sev = params['hosp_rt'].value - params['mort_rt'].value
    p_recov_mild = 1 - p_fatal - p_recov_sev

    ## FLOWS ##
    dSdt = -min(_beta * I[-1], S[-1])
    dEdt = min(_beta * I[-1], S[-1]) - _sigma * E[-1]
    E_newdt = min(_beta * I[-1], S[-1])

    dI_Milddt = p_recov_mild * _sigma * E[-1] - _gamma * I_Mild[-1]
    dI_Sevdt = p_recov_sev * _sigma * E[-1] - _nu * I_Sev[-1]
    dI_Fataldt = p_fatal * _sigma * E[-1] - _nu * I_Fatal[-1]

    dH_Sevdt = _nu * I_Sev[-1] - _rho * H_Sev[-1]
    dH_Fataldt = _nu * I_Fatal[-1] - _mu * H_Fatal[-1]
    H_Admitsdt = _nu * I_Sev[-1] + _nu * I_Fatal[-1]

    dRdt = _gamma * I_Mild[-1] + _rho * H_Sev[-1]
    dDdt = _mu * H_Fatal[-1]

    ## LEVELS ##
    S.append(S[-1] + dSdt)

    E.append(E[-1] + dEdt)
    E_new.append(E_newdt)

    I_Mild.append(I_Mild[-1] + dSdt)
    I_Sev.append(I_Sev[-1] + dSdt)
    I_Fatal.append(I_Fatal[-1] + dSdt)
    I.append(I_Mild[-1] + I_Sev[-1] + I_Fatal[-1])

    H_Sev.append(H_Sev[-1] + dH_Sevdt)
    H_Fatal.append(H_Fatal[-1] + dH_Fataldt)
    H_Admits.append(H_Admitsdt)

    R.append(R[-1] + dRdt)
    D.append(D[-1] + dDdt)

    prop_recovered = R[-1] / (R[-1] + S[-1])
    # ^ Can be thought of as the prob that a vaccine is going to a person with antibodies
    next_vaxxed = 0 if np.isnan(vax_new_people_daily.shift(28).loc[cohort_strt]) else \
    vax_new_people_daily.shift(28).loc[cohort_strt]
    next_recovered_vaxxed = prop_recovered * next_vaxxed
    next_susceptible_vaxxed = (1 - prop_recovered) * next_vaxxed
    # print(f'cohort_st: {cohort_strt} suspop: {suspop[-1]} dS: {dS}, next_susceptible_vaxxed: {next_susceptible_vaxxed}')
    next_suspop = max(suspop[-1] + dS - (next_susceptible_vaxxed), 0)

    recovered_vaxxed.append(recovered_vaxxed[-1] + next_recovered_vaxxed)
    susceptible_vaxxed.append(susceptible_vaxxed[-1] + next_susceptible_vaxxed)
    suspop.append(next_suspop)

    totpopchk = df_agg.loc[cohort_strt, ['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths']].sum()

    # if (round(totpopchk + suspop[-1]) != round(model_dict['tot_pop'])):
    if abs((totpopchk + suspop[-1] + susceptible_vaxxed[-1]) / model_dict['tot_pop'] - 1) > 1e-4:
        print(df_all_cohorts)
        print(df_all_cohorts.sum(axis=1).unstack())
        print(cohort_strt)
        print('totpop: ', round(model_dict['tot_pop']))
        print('dS ', dS)
        print('sum of df_agg', totpopchk)
        print(f'suspop[-1]: {suspop[-1]} susceptible_vaxxed[-1]: {susceptible_vaxxed[-1]}')
        print('sum of both', round(totpopchk + suspop[-1] + susceptible_vaxxed[-1]))
        raise Exception('Agg total population varies by more than 0.01%')

model_dict['df_rts']['rt_scenario'] = r_t.iloc[:-1]
model_dict['df_rts']['rt_nochange'] = r_t_nochange.iloc[:-1]
model_dict['df_rts']['rt_preimmune'] = r_t_preimmune.iloc[:-1]

df_agg = df_all_cohorts.sum(axis=1).unstack()
df_agg.index = pd.DatetimeIndex(df_agg.index).normalize()

list2series_dt_idx = pd.date_range(
    start_dt - pd.Timedelta(days=1),
    start_dt + pd.Timedelta(days=model_dict['d_to_forecast'] - 1))

df_agg['susceptible'] = pd.Series(suspop, index=list2series_dt_idx)
df_agg['vaccinated_prev_infected'] = pd.Series(recovered_vaxxed, index=list2series_dt_idx)
df_agg['vaccinated_never_infected'] = pd.Series(susceptible_vaxxed, index=list2series_dt_idx)
df_agg['recovered_unvaccinated'] = df_agg['recovered'] - df_agg['vaccinated_prev_infected']

exposed_daily = df_all_cohorts.stack().unstack(['metric'])[['exposed']].reset_index()
df_agg['exposed_daily'] = exposed_daily[(exposed_daily.dt == exposed_daily.cohort_dt)].set_index(['dt'])['exposed']
df_agg['deaths_daily'] = df_agg['deaths'].diff()

df_agg['hospitalized_fitted'] = df_agg['hospitalized']
df_agg['hospitalized'] = lvl_adj_forecast(model_dict['df_hist']['hosp_concur'], df_agg['hospitalized'])
df_agg['deaths_fitted'] = df_agg['deaths']
df_agg['deaths'] = lvl_adj_forecast(model_dict['df_hist']['deaths_tot'], df_agg['deaths'])

model_dict['df_agg'] = df_agg.dropna()
model_dict['df_all_cohorts'] = df_all_cohorts