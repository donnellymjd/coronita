# start_dt = pd.Timestamp('2020-02-01')
# model_dict = allstate_model_dicts['NJ']
# exposed_0 = 100
# infectious_0 = 0
# pg_summer_dt = '2021-07-04'
#
# # def seir_model_cohort(start_dt, model_dict, exposed_0=100, infectious_0=100):
#
# vax_new_people_daily = model_dict['df_hist']['vax_initiated'].diff() \
#     .reindex(model_dict['df_vax_fore'].index).fillna(model_dict['df_vax_fore']['trend'].diff())
# vax_new_people_daily = vax_new_people_daily.reindex(index=pd.date_range(start_dt, vax_new_people_daily.index[-1]))
# vax_new_people_daily = vax_new_people_daily.fillna(method='bfill')
#
# V_Sus_0 = model_dict['df_vax_fore']['trend'].loc[start_dt] \
#     if start_dt in model_dict['df_vax_fore']['trend'].index else 0
#
# ## Initial Conditions ##
# N = model_dict['tot_pop']
# E = [exposed_0]
# E_new = [exposed_0]
# I_Mild = [infectious_0]
# I_Sev = [infectious_0]
# I_Fatal = [infectious_0]
# H_Sev = [0]
# H_Fatal = [0]
# H_Admits = [0]
# R = [0]
# D = [0]
# V_Sus = [V_Sus_0]
# V_Rec = [0]
# S = [N - E[-1] - V_Sus[-1]]
#
# t = np.linspace(0, model_dict['d_to_forecast'], model_dict['d_to_forecast'] + 1)
#
# r_t = pd.Series(np.nan,
#                 index=pd.date_range(start_dt,
#                                     max(start_dt + pd.Timedelta(days=model_dict['d_to_forecast']),
#                                         pd.Timestamp(pg_summer_dt))
#                                     )
#                 )
#
# if 'rt_scenario' in model_dict['df_rts'].columns:
#     r_t = r_t.fillna(model_dict['df_rts']['rt_scenario']).fillna(method='bfill').fillna(method='ffill')
# else:
#     local_r0_date = model_dict['df_rts'].loc['2020-02-01':'2020-04-30', 'weighted_average'].idxmax()
#     # print('local_r0_date: ', local_r0_date)
#     r_t = r_t.fillna(model_dict['df_rts'].loc[local_r0_date:, 'weighted_average'])
#     r_t = r_t.fillna(method='bfill').fillna(method='ffill')
# r_t_preimmune = r_t.copy()
# # print(r_t.last_valid_index())
#
# last_obs_rt = model_dict['df_rts']['weighted_average'].last_valid_index()
#
# model_dict['df_rts'] = model_dict['df_rts'].reindex(r_t.index)
# model_dict['df_rts']['policy_triggered'] = 0
# model_dict['hosp_cap_dt'] = None
#
# last_r = r_t.iloc[0]
#
# for t_ in t[:-1]:
#     cohort_strt = start_dt + pd.Timedelta(days=t_)
#     # print(f'cohort_strt: {cohort_strt}, next_suspop: {next_suspop}')
#
#     if (model_dict['covid_params']['policy_trigger']
#             and (cohort_strt > last_obs_rt)):
#
#         if 'hosp_beds_avail' in model_dict['df_hist'].columns:
#             covid_hosp_capacity = \
#                 model_dict['df_hist']['hosp_beds_avail'].replace(0, np.nan).rolling(7).mean().dropna().iloc[-1]
#             covid_hosp_capacity = covid_hosp_capacity + model_dict['df_hist']['hosp_concur'].dropna().iloc[-1]
#         else:
#             tot_hosp_capacity = model_dict['tot_pop'] / 1000 * 2.7
#             covid_hosp_capacity = tot_hosp_capacity * 0.2
#
#         if (((H_Sev[-1] + H_Fatal[-1]) > covid_hosp_capacity)
#                 or (model_dict['covid_params']['policy_trigger_once']
#                     and model_dict['df_rts']['policy_triggered'].sum() > 1)):
#             r_t.loc[cohort_strt] = 0.9
#             model_dict['df_rts'].loc[cohort_strt, 'policy_triggered'] = 1
#             if model_dict['hosp_cap_dt'] == None:
#                 model_dict['hosp_cap_dt'] = cohort_strt
#
#     # ACCOUNT FOR EFFECT OF IMMUNITY IN FORECAST PERIOD #
#     # Math:
#     # rt / (suspop_lastdayofobs / tot_pop) * (suspop_t / tot_pop)
#     # rt * (tot_pop / suspop_lastdayofobs) * (suspop_t / tot_pop)
#     # # tot_pop cancels
#     # rt * suspop_t / suspop_lastdayofobs
#     if cohort_strt <= last_obs_rt:
#         r_t_preimmune.loc[cohort_strt] = r_t.loc[cohort_strt] / (
#                 S[-1] / model_dict['tot_pop']) if S[-1] > 0 else 1.0
#
#     if cohort_strt == last_obs_rt:
#         suspop_lastdayofobs = S[-1] if S[-1] > 0 else 1.0
#         # r_t_preimmune = r_t.div(suspop_lastdayofobs / model_dict['tot_pop'])
#         r_t_preimmune.loc[cohort_strt + pd.Timedelta(days=1):] = np.nan
#
#         current_r0 = model_dict['covid_params']['voc_transmissibility'] * model_dict['covid_params']['basic_r0']
#         current_r0 = max(min(current_r0, 3.0), 2.0)
#         model_dict['covid_params']['current_r0'] = current_r0
#         # print(f'current_r0 {current_r0}')
#         if r_t_preimmune.loc[r_t_preimmune.last_valid_index()] > current_r0:
#             r_t_preimmune.loc[pg_summer_dt:] = r_t_preimmune.loc[r_t_preimmune.last_valid_index()]
#         else:
#             r_t_preimmune.loc[pg_summer_dt:] = current_r0
#         r_t_preimmune = r_t_preimmune.interpolate()
#         r_t_nochange = r_t.copy()
#         # if pg_summer_dt in r_t_preimmune.index:
#         #     print(r_t_preimmune.loc[cohort_strt], r_t_preimmune.loc[pg_summer_dt])
#         # print(f'{cohort_strt} next_suspop: {next_suspop}')
#     elif cohort_strt > last_obs_rt:
#         r_t_nochange.loc[cohort_strt] = r_t_nochange.loc[cohort_strt] * (S[-1] / suspop_lastdayofobs)
#         r_t.loc[cohort_strt] = r_t_preimmune.loc[cohort_strt] * (S[-1] / N)
#
#     this_r = r_t.loc[cohort_strt]
#
#     if this_r != last_r:
#         last_r = this_r
#
#     params = model_dict['covid_params']
#
#     ## RATES ##
#     _sigma = 1 / params['d_incub']
#     _gamma = 1 / params['d_infect']
#     _beta = this_r * _gamma
#     _nu = 1 / params['d_to_hosp']
#     _rho = 1 / params['d_in_hosp']
#     _mu = 1 / params['d_til_death']
#     p_fatal = params['mort_rt']
#
#     #     if cohort_strt in model_dict['df_agg']['deaths_daily'].dropna().index:
#     #         if model_dict['df_agg'].loc[cohort_strt, 'deaths_daily'] > 0 :
#     #             mort_mult = model_dict['df_agg'].loc[cohort_strt, 'deaths_daily'] / (_mu * H_Fatal[-1])
#     #             p_fatal = p_fatal * mort_mult
#     p_fatal = 0.025 if cohort_strt < pd.Timestamp('2020-07-01') else params['mort_rt']
#
#     if p_fatal <= 0:
#         print(t_, p_fatal)
#     p_recov_sev = params['hosp_rt'] - p_fatal
#     print(f'p_fatal: {p_fatal}, p_recov_sev: {p_recov_sev}')
#     p_recov_mild = 1 - p_fatal - p_recov_sev
#     prop_recovered = max(min(R[-1] / (R[-1] + S[-1]), 1), 0)
#     # ^ Can be thought of as the prob that a vaccine is going to a person with antibodies
#
#     ## FLOWS ##
#
#     dVdt = 0 if np.isnan(vax_new_people_daily.shift(28).loc[cohort_strt]) else \
#         vax_new_people_daily.shift(28).loc[cohort_strt]
#     dV_Recdt = prop_recovered * dVdt
#     dV_Susdt = (1 - prop_recovered) * dVdt
#
#     E_newdt = min(_beta * (I_Mild[-1] + I_Sev[-1] + I_Fatal[-1]), S[-1] - dV_Susdt)
#     dSdt = -E_newdt - dV_Susdt
#     dEdt = E_newdt - _sigma * E[-1]
#
#     dI_Milddt = p_recov_mild * _sigma * E[-1] - _gamma * I_Mild[-1]
#     dI_Sevdt = p_recov_sev * _sigma * E[-1] - _nu * I_Sev[-1]
#     dI_Fataldt = p_fatal * _sigma * E[-1] - _nu * I_Fatal[-1]
#
#     dH_Sevdt = _nu * I_Sev[-1] - _rho * H_Sev[-1]
#     dH_Fataldt = _nu * I_Fatal[-1] - _mu * H_Fatal[-1]
#     H_Admitsdt = _nu * I_Sev[-1] + _nu * I_Fatal[-1]
#
#     R_newdt = _gamma * I_Mild[-1] + _rho * H_Sev[-1]
#     dRdt = R_newdt - dV_Recdt
#     dDdt = _mu * H_Fatal[-1]
#
#     ## LEVELS ##
#     S.append(S[-1] + dSdt)
#
#     E.append(E[-1] + dEdt)
#     E_new.append(E_newdt)
#
#     I_Mild.append(I_Mild[-1] + dI_Milddt)
#     I_Sev.append(I_Sev[-1] + dI_Sevdt)
#     I_Fatal.append(I_Fatal[-1] + dI_Fataldt)
#
#     H_Sev.append(H_Sev[-1] + dH_Sevdt)
#     H_Fatal.append(H_Fatal[-1] + dH_Fataldt)
#     H_Admits.append(H_Admitsdt)
#
#     R.append(R[-1] + dRdt)
#     D.append(D[-1] + dDdt)
#     V_Sus.append(V_Sus[-1] + dV_Susdt)
#     V_Rec.append(V_Sus[-1] + dV_Recdt)
#
# df_agg = pd.DataFrame(
#     [S, E, E_new, I_Mild, I_Sev, I_Fatal, H_Sev, H_Fatal, H_Admits, R, D, V_Sus, V_Rec]
# ).T
# df_agg.index = pd.date_range(
#     start_dt - pd.Timedelta(days=1),
#     start_dt + pd.Timedelta(days=model_dict['d_to_forecast'] - 1))
# df_agg.columns = ['susceptible', 'exposed', 'exposed_daily',
#                   'I_Mild', 'I_Sev', 'I_Fatal',
#                   'H_Sev', 'H_Fatal', 'hosp_admits',
#                   'recovered_unvaccinated', 'deaths',
#                   'vaccinated_never_infected', 'vaccinated_prev_infected',
#                   ]
# df_agg['recovered'] = df_agg[['recovered_unvaccinated', 'vaccinated_prev_infected']].sum(axis=1)
# df_agg['infectious'] = df_agg[['I_Mild', 'I_Sev', 'I_Fatal']].sum(axis=1)
# df_agg['hospitalized'] = df_agg[['H_Sev', 'H_Fatal']].sum(axis=1)
# df_agg['icu'] = np.nan
# df_agg['vent'] = np.nan
# df_agg['deaths_daily'] = df_agg['deaths'].diff()

import pandas as pd
import numpy as np

def vax_diff_model(t_, start_dt, model_dict, level_lists, l_rts, vax_new_people_daily):
    cohort_strt = start_dt + pd.Timedelta(days=t_)
    S, E, E_new, I_Mild, I_Sev, I_Fatal, H_Sev, H_Fatal, H_Admits, R, D, V_Sus, V_Rec = level_lists
    r_t, r_t_nochange, r_t_preimmune, last_r, last_obs_rt = l_rts
    pg_summer_dt = model_dict['covid_params']['pg_summer_dt']

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

        if (((H_Sev[-1] + H_Fatal[-1]) > covid_hosp_capacity)
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
                S[-1] / model_dict['tot_pop']) if S[-1] > 0 else 1.0



    if cohort_strt == last_obs_rt:
        suspop_lastdayofobs = S[-1] if S[-1] > 0 else 1.0
        # r_t_preimmune = r_t.div(suspop_lastdayofobs / model_dict['tot_pop'])
        r_t_preimmune.loc[cohort_strt + pd.Timedelta(days=1):] = np.nan

        current_r0 = model_dict['covid_params']['voc_transmissibility'] * model_dict['covid_params']['basic_r0']
        current_r0 = max(min(current_r0, 3.0), 2.0)
        model_dict['covid_params']['current_r0'] = current_r0
        # print(f'current_r0 {current_r0}')
        if r_t_preimmune.loc[r_t_preimmune.last_valid_index()] > current_r0:
            r_t_preimmune.loc[pg_summer_dt:] = r_t_preimmune.loc[r_t_preimmune.last_valid_index()]
        else:
            r_t_preimmune.loc[pg_summer_dt:] = current_r0
        r_t_preimmune = r_t_preimmune.interpolate()
        r_t_nochange = r_t.copy()
        # if pg_summer_dt in r_t_preimmune.index:
        #     print(r_t_preimmune.loc[cohort_strt], r_t_preimmune.loc[pg_summer_dt])
        # print(f'{cohort_strt} next_suspop: {next_suspop}')
    elif cohort_strt > last_obs_rt:
        idx_lastdayofobs = (last_obs_rt - start_dt).days
        suspop_lastdayofobs = S[idx_lastdayofobs]
        r_t_nochange.loc[cohort_strt] = r_t_nochange.loc[cohort_strt] * (S[-1] / suspop_lastdayofobs)
        r_t.loc[cohort_strt] = r_t_preimmune.loc[cohort_strt] * (S[-1] / model_dict['tot_pop'])

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

    #     if cohort_strt in model_dict['df_agg']['deaths_daily'].dropna().index:
    #         if model_dict['df_agg'].loc[cohort_strt, 'deaths_daily'] > 0 :
    #             mort_mult = model_dict['df_agg'].loc[cohort_strt, 'deaths_daily'] / (_mu * H_Fatal[-1])
    #             p_fatal = p_fatal * mort_mult
    p_fatal = 0.025 if cohort_strt < pd.Timestamp('2020-07-01') else params['mort_rt']

    if p_fatal <= 0:
        print(t_, p_fatal)
    p_recov_sev = params['hosp_rt'] - p_fatal
    # print(f'p_fatal: {p_fatal}, p_recov_sev: {p_recov_sev}')
    p_recov_mild = 1 - p_fatal - p_recov_sev
    prop_recovered = max(min(R[-1] / (R[-1] + S[-1]), 1), 0)
    # ^ Can be thought of as the prob that a vaccine is going to a person with antibodies

    ## FLOWS ##

    dVdt = 0 if np.isnan(vax_new_people_daily.shift(28).loc[cohort_strt]) else \
        vax_new_people_daily.shift(28).loc[cohort_strt]
    dV_Recdt = prop_recovered * dVdt
    dV_Susdt = (1 - prop_recovered) * dVdt

    E_newdt = min(_beta * (I_Mild[-1] + I_Sev[-1] + I_Fatal[-1]), S[-1] - dV_Susdt)
    dSdt = -E_newdt - dV_Susdt
    dEdt = E_newdt - _sigma * E[-1]

    dI_Milddt = p_recov_mild * _sigma * E[-1] - _gamma * I_Mild[-1]
    dI_Sevdt = p_recov_sev * _sigma * E[-1] - _nu * I_Sev[-1]
    dI_Fataldt = p_fatal * _sigma * E[-1] - _nu * I_Fatal[-1]

    dH_Sevdt = _nu * I_Sev[-1] - _rho * H_Sev[-1]
    dH_Fataldt = _nu * I_Fatal[-1] - _mu * H_Fatal[-1]
    H_Admitsdt = _nu * I_Sev[-1] + _nu * I_Fatal[-1]

    R_newdt = _gamma * I_Mild[-1] + _rho * H_Sev[-1]
    dRdt = R_newdt - dV_Recdt
    dDdt = _mu * H_Fatal[-1]

    output_diffs = [dVdt, dV_Recdt, dV_Susdt, E_newdt, dSdt, dEdt,
                    dI_Milddt, dI_Sevdt, dI_Fataldt,
                    dH_Sevdt, dH_Fataldt, H_Admitsdt,
                    R_newdt, dRdt, dDdt]
    output_rts = r_t, r_t_nochange, r_t_preimmune, last_r, last_obs_rt

    return (model_dict, output_diffs, output_rts)

def append_model_diffs(input_diffs, level_lists):
    dVdt, dV_Recdt, dV_Susdt, E_newdt, dSdt, dEdt, \
        dI_Milddt, dI_Sevdt, dI_Fataldt, \
        dH_Sevdt, dH_Fataldt, H_Admitsdt, \
        R_newdt, dRdt, dDdt = input_diffs

    S, E, E_new, I_Mild, I_Sev, I_Fatal, \
        H_Sev, H_Fatal, H_Admits, R, D, V_Sus, V_Rec = level_lists

    ## LEVELS ##
    S.append(S[-1] + dSdt)

    E.append(E[-1] + dEdt)
    E_new.append(E_newdt)

    I_Mild.append(I_Mild[-1] + dI_Milddt)
    I_Sev.append(I_Sev[-1] + dI_Sevdt)
    I_Fatal.append(I_Fatal[-1] + dI_Fataldt)

    H_Sev.append(H_Sev[-1] + dH_Sevdt)
    H_Fatal.append(H_Fatal[-1] + dH_Fataldt)
    H_Admits.append(H_Admitsdt)

    R.append(R[-1] + dRdt)
    D.append(D[-1] + dDdt)
    V_Sus.append(V_Sus[-1] + dV_Susdt)
    V_Rec.append(V_Sus[-1] + dV_Recdt)
    output_lists = [S, E, E_new, I_Mild, I_Sev, I_Fatal,
                    H_Sev, H_Fatal, H_Admits, R, D, V_Sus, V_Rec]

    return output_lists