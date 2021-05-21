import pandas as pd
import numpy as np
from scipy.stats import gamma
from prophet import Prophet
from statsmodels.stats.weightstats import DescrStatsW
# from coronita_chart_helper import *

def outlier_removal(raw_series, num_std=3):
    rolling_avg = raw_series.rolling(7, center=True, min_periods=3).mean().fillna(method='bfill').fillna(method='ffill')
    noise = raw_series - rolling_avg
    cleaned_series = raw_series[(np.abs(noise) < (noise.std() * num_std))].reindex(raw_series.index)

    outlier_idx = raw_series[(np.abs(noise) >= (noise.std() * num_std))].index
    if outlier_idx.shape[0] > 0:
        before_first_outlier = raw_series.loc[:outlier_idx[0]].iloc[:-1]
        after_last_outlier = raw_series.loc[outlier_idx[-1]:].iloc[1:]

        if ((before_first_outlier.shape[0] > 1)
                and (before_first_outlier.sum() == 0)
                and (before_first_outlier.std() == 0)):
            cleaned_series.loc[:outlier_idx[0]] = np.nan

        if ((after_last_outlier.shape[0] > 1)
                and (after_last_outlier.sum() == 0)
                and (after_last_outlier.std() == 0)):
            cleaned_series.loc[:outlier_idx[0]] = np.nan

    firstnonzero = cleaned_series.replace(0, np.nan).first_valid_index()
    cleaned_series.loc[:firstnonzero] = cleaned_series.replace(0, np.nan)
    lastnonzero = cleaned_series.replace(0, np.nan).first_valid_index()
    cleaned_series.loc[lastnonzero:] = cleaned_series.replace(0, np.nan)

    return cleaned_series

def lvl_adj_forecast(s_hist_lvl, s_fore_lvl): #lvl_adj_forecast(model_dict, hist_lvl_name, fore_lvl_name):
    # df_agg = model_dict['df_agg'].copy()
    # hist_last_day = model_dict['df_hist'][hist_lvl_name].last_valid_index()
    # fore_diff_future = df_agg[fore_lvl_name].diff().loc[hist_last_day + pd.Timedelta(days=1):]
    # fore_lvl_adjusted = fore_diff_future.cumsum().add(model_dict['df_hist'][hist_lvl_name].loc[hist_last_day])
    #
    # df_agg[fore_lvl_name+'_fitted'] = df_agg[fore_lvl_name]
    # df_agg[fore_lvl_name] = fore_lvl_adjusted
    # df_agg[fore_lvl_name] = df_agg[fore_lvl_name].fillna(df_agg[fore_lvl_name+'_fitted'])
    # model_dict['df_agg'] = df_agg
    # return model_dict
    s_hist_lvl = s_hist_lvl.rolling(7).mean()
    hist_last_day = s_hist_lvl.last_valid_index()
    fore_diff_future = s_fore_lvl.diff().loc[hist_last_day + pd.Timedelta(days=1):]
    jumping_off_scalar = (s_hist_lvl.loc[hist_last_day] / s_fore_lvl.loc[hist_last_day])
    fore_diff_future = fore_diff_future * jumping_off_scalar
    fore_lvl_adjusted = fore_diff_future.cumsum().add(s_hist_lvl.loc[hist_last_day])
    fore_lvl_adjusted = fore_lvl_adjusted.reindex(s_fore_lvl.index)
    fore_lvl_adjusted = fore_lvl_adjusted.fillna(s_fore_lvl)

    return fore_lvl_adjusted

def normal_hosp_cap(model_dict):
    covid_hosp_capacity = model_dict['df_hist']['hosp_beds_avail'].replace(0, np.nan).dropna()
    covid_hosp_capacity = covid_hosp_capacity + model_dict['df_hist']['hosp_concur'].dropna()
    covid_hosp_capacity = covid_hosp_capacity.rolling(7).mean()
    normal_capacity = covid_hosp_capacity.quantile(.05)
    return normal_capacity

def daily_cohort_model(cohort_strt, d_to_fore, covid_params, E_0, I_0=0):
    t = np.linspace(0, int(d_to_fore) - 1, int(d_to_fore))

    E = [E_0]
    I = [I_0]
    H = [0]
    ICU = [0]
    R = [0]
    D = [0]
    H_inflow = [0]

    norm_fact_p_dI = pd.Series(np.arange(covid_params['d_incub'] * 10)).apply(gamma.pdf, a=covid_params['d_incub']).sum()
    norm_fact_p_dmR = pd.Series(
        np.arange((covid_params['d_infect'] + covid_params['d_incub']) * 10)).apply(
        gamma.pdf, a=(covid_params['d_infect'] + covid_params['d_incub'])).sum()

    for t_ in t[:-1]:

        #### PROBABILITY DISTRIBUTIONS FOR FLOWS ####
        prob_dI_t = gamma.pdf(t_, covid_params['d_incub'])
        prob_dI_t = prob_dI_t / norm_fact_p_dI

        prob_mild_dR_t = (1 - covid_params['hosp_rt']) * gamma.pdf(t_, covid_params['d_infect'] + covid_params['d_incub'])
        prob_mild_dR_t = prob_mild_dR_t / norm_fact_p_dmR

        prob_H_inflow_fromE0_t = covid_params['hosp_rt'] * gamma.pdf(t_, covid_params['d_to_hosp'] + covid_params['d_incub'])
        prob_H_inflow_fromI0_t = covid_params['hosp_rt'] * gamma.pdf(t_, covid_params['d_to_hosp'] / 2, scale=2)

        prob_sev_dR_t = ( (covid_params['hosp_rt'] - covid_params['mort_rt'])
                          * gamma.pdf(t_, (covid_params['d_incub'] + covid_params['d_in_hosp'] + covid_params['d_to_hosp']) / 4, scale=4) )

        prob_dD_t = covid_params['mort_rt'] * gamma.pdf(t_, (covid_params['d_til_death'] + covid_params['d_incub']) / 1, scale=1)
        #############################################

        ############## FLOW ACCOUNTING ##############
        # Change in Exposed Population
        dE = -1 * min(prob_dI_t * E[0], E[-1])

        # Hospital Outflows, limited to be no more than hospital capacity
        d_hosp_outflow = -1 * min((prob_sev_dR_t * (E[0] + I[0])
                                   + H[0] * (prob_sev_dR_t / (covid_params['hosp_rt'] - covid_params['mort_rt'])) * (1 - covid_params['mort_rt'] / covid_params['hosp_rt']))
                                  + (prob_dD_t * (E[0] + I[0])
                                     + H[0] * (prob_dD_t / covid_params['mort_rt']) * covid_params['mort_rt'] / covid_params['hosp_rt']),
                                  H[-1])

        if (prob_sev_dR_t + prob_dD_t) > 0:
            # Severe Recoveries - Component of Hospital Outflows
            d_sevR = (-1 * prob_sev_dR_t * d_hosp_outflow) / (prob_sev_dR_t + prob_dD_t)

            # Deaths - Component of Hospital Outflows
            dD = (-1 * prob_dD_t * d_hosp_outflow) / (prob_sev_dR_t + prob_dD_t)
        else:
            # Severe Recoveries - Component of Hospital Outflows
            d_sevR = 0.
            # Deaths - Component of Hospital Outflows
            dD = 0.

        # New Hospital Admittances
        d_hosp_admits = prob_H_inflow_fromE0_t * E[0] + prob_H_inflow_fromI0_t * I[0]

        # Infectious Inflows
        dI_inflow = -1 * dE

        # Mild Recoveries
        d_mildR = min(prob_mild_dR_t * (E[0] + I[0]), I[-1] + dI_inflow - d_hosp_admits)

        # Change in Recovered Population
        dR = d_mildR + d_sevR

        # Change in Hospitalized Population
        dH = d_hosp_admits + d_hosp_outflow

        # Infectious Outflows
        dI_outflow = d_mildR + d_hosp_admits

        # Net change in Infectious Population
        dI = dI_inflow - dI_outflow

        if round(dI_inflow - dI_outflow) < round(-1 * I[-1]):
            print('dI_inflow', dI_inflow)
            print('dI_outflow', dI_outflow)
            print('round(dI_inflow - dI_outflow)', round(dI_inflow - dI_outflow))
            print('I[-1]', I[-1])
            raise Exception(cohort_strt, 'Daily Cohort Infectious Net Outflows are greater than Infectious Population')
            #############################################

        E.append(E[-1] + dE)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
        H.append(H[-1] + dH)
        D.append(D[-1] + dD)
        H_inflow.append(d_hosp_admits)
    df_out = pd.DataFrame(np.stack([E, I, R, H, D, H_inflow]).T,
                          columns=['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths', 'hosp_admits'],
                          index=pd.date_range(cohort_strt,
                                              cohort_strt + pd.Timedelta(days=d_to_fore - 1)))
    df_out['icu'] = df_out['hospitalized'].mul(covid_params['icu_rt'])
    df_out['vent'] = df_out['hospitalized'].mul(covid_params['icu_rt'] * covid_params['vent_rt'])
    df_out.index = pd.DatetimeIndex(df_out.index).normalize()
    df_out.index.name = 'dt'
    df_out.columns.name = 'metric'

    return df_out

def seir_model_cohort(start_dt, model_dict, exposed_0=100, infectious_0=100):
    next_infectious = infectious_0
    next_hospitalized = 0
    _gamma = 1 / (model_dict['covid_params']['d_infect'])

    vax_new_people_daily = model_dict['df_hist']['vax_initiated'].diff() \
        .reindex(model_dict['df_vax_fore'].index).fillna(model_dict['df_vax_fore']['trend'].diff())
    vax_new_people_daily = vax_new_people_daily.reindex(index=pd.date_range(start_dt, vax_new_people_daily.index[-1]))
    vax_new_people_daily = vax_new_people_daily.fillna(method='bfill')
    next_recovered_vaxxed = 0
    next_susceptible_vaxxed = vax_new_people_daily.loc[start_dt] if start_dt in vax_new_people_daily.index else 0
    next_suspop = model_dict['tot_pop'] - exposed_0 - infectious_0 - next_susceptible_vaxxed

    recovered_vaxxed = [next_recovered_vaxxed]
    susceptible_vaxxed = [next_susceptible_vaxxed]
    suspop = [next_suspop]

    t = np.linspace(0, model_dict['d_to_forecast'], model_dict['d_to_forecast'] + 1)

    df_all_cohorts = pd.DataFrame()
    df_all_cohorts.columns.name = 'cohort_dt'

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
                and (cohort_strt > last_obs_rt) ):

            if 'hosp_beds_avail' in model_dict['df_hist'].columns:
                covid_hosp_capacity = model_dict['df_hist']['hosp_beds_avail'].replace(0,np.nan).rolling(7).mean().dropna().iloc[-1]
                covid_hosp_capacity = covid_hosp_capacity + model_dict['df_hist']['hosp_concur'].dropna().iloc[-1]
            else:
                tot_hosp_capacity = model_dict['tot_pop']/1000 * 2.7
                covid_hosp_capacity = tot_hosp_capacity * 0.2

            if ( (next_hospitalized > covid_hosp_capacity)
                    or (model_dict['covid_params']['policy_trigger_once']
                        and model_dict['df_rts']['policy_triggered'].sum() > 1) ):
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
        beta = this_r * _gamma

        d_to_fore = t[-1] - t_ + 1

        if this_r != last_r:
            last_r = this_r

        if t_ == 0:
            dS = 0

            df_daily_cohort = daily_cohort_model(
                cohort_strt, d_to_fore,
                model_dict['covid_params'], E_0=exposed_0, I_0=infectious_0)

            df_all_cohorts[cohort_strt] = df_daily_cohort.stack()

            df_daily_cohort_scalar = daily_cohort_model(
                cohort_strt, d_to_fore,
                model_dict['covid_params'], E_0=1e6, I_0=0).reset_index(drop=True)

        else:
            # dS = -1 * min(beta * suspop[-1] * next_infectious / model_dict['tot_pop'], suspop[-1])
            dS = -1 * min(beta * next_infectious, suspop[-1])

            df_daily_cohort = dS * -1 * df_daily_cohort_scalar.iloc[:int(d_to_fore)] / 1e6

            df_daily_cohort.index = pd.date_range(cohort_strt,
                                                  cohort_strt + pd.Timedelta(days=d_to_fore - 1))
            df_all_cohorts[cohort_strt] = df_daily_cohort.stack()

        d_cohort_totpop_std = round(df_daily_cohort[
                                        ['exposed', 'deaths', 'hospitalized', 'infectious', 'recovered']
                                    ].dropna().sum(axis=1).std(), 1)

        if d_cohort_totpop_std != 0.0:
            print(cohort_strt, ' rt: ', this_r)
            print(f'next suspop: {next_suspop}, prop_recovered: {prop_recovered}, next_vaxxed: {next_vaxxed}')
            print(f'recovered: {df_agg.loc[cohort_strt, "recovered"]} next_susceptible_vaxxed: {next_susceptible_vaxxed}')
            print(cohort_strt, d_cohort_totpop_std)
            print(df_daily_cohort)
            raise Exception('Daily Cohort total population varies significantly')

        df_agg = df_all_cohorts.sum(axis=1).unstack()
        df_agg.index = pd.DatetimeIndex(df_agg.index).normalize()

        # VACCINE IMPACT #
        # if 'vaccine_prop_t' in model_dict.keys():
        #     if cohort_strt - pd.Timedelta(days=7) < model_dict['vaccine_prop_t'].diff().first_valid_index():
        #         newly_vaccinated = 0
        #     else:
        #         newly_vaccinated = model_dict['vaccine_prop_t'].diff().loc[cohort_strt - pd.Timedelta(days=7)] * model_dict['tot_pop']
        #     # We need to make an adjustment for the fact that some people in the recovered population are also getting vaccinated.
        #     prop_recovered = df_agg.loc[cohort_strt, 'recovered'] / model_dict['tot_pop']
        #     prop_vaxxed = model_dict['vaccine_prop_t'].loc[cohort_strt - pd.Timedelta(days=1)]
        #     # Proportion of population recovered by not yet vaccinated...
        #     prop_recovered_unvax = prop_recovered - prop_vaxxed
        #     new_justvax_recovered = newly_vaccinated * (1 - prop_recovered_unvax)
        # else:
        #     new_justvax_recovered = 0
        # idx = pd.IndexSlice
        # df_all_cohorts.loc[idx[:, ['recovered']], cohort_strt] = df_all_cohorts.loc[idx[:, ['recovered']], cohort_strt] + new_justvax_recovered

        next_infectious = df_agg.loc[cohort_strt, 'infectious']
        # next_hospitalized = df_agg.loc[cohort_strt, 'hospitalized']
        next_hospitalized = lvl_adj_forecast(model_dict['df_hist']['hosp_concur'], df_agg['hospitalized']).loc[cohort_strt]
        # next_suspop = max(suspop[-1] + dS, 0)

        next_recovered = 0 if np.isnan(df_agg.loc[cohort_strt, 'recovered']) else df_agg.loc[cohort_strt, 'recovered']
        prop_recovered = next_recovered / (next_recovered + suspop[-1] + dS)
        # ^ Can be thought of as the prob that a vaccine is going to a person with antibodies
        next_vaxxed = 0 if np.isnan(vax_new_people_daily.shift(28).loc[cohort_strt]) else vax_new_people_daily.shift(28).loc[cohort_strt]
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

    return model_dict

def fore_rmse(obs_metric, pred_metric):
    from sklearn.metrics import mean_squared_error
    df_compare = pd.DataFrame()

    df_compare['obs_metric'] = obs_metric
    df_compare['pred_metric'] = pred_metric

    df_compare = df_compare.dropna()
    df_compare = df_compare.iloc[-60:]
    rmse = np.sqrt(mean_squared_error(df_compare['obs_metric'], df_compare['pred_metric']))
    # norm_rmse = rmse / (df_compare['obs_metric'].max() - df_compare['obs_metric'].min()) * 1e3
    norm_rmse = rmse / df_compare['obs_metric'].mean() * 1e3
    avg_error = df_compare['obs_metric'].sub(df_compare['pred_metric']).mean()
    rel_error = df_compare['obs_metric'].div(df_compare['pred_metric']).mean()
    return pd.Series([norm_rmse, avg_error, rel_error], index=['rmse', 'avg_error', 'rel_error'])

def model_find_start(this_guess, model_dict, exposed_0=None, infectious_0=None):
    from sklearn.metrics import mean_squared_error
    # from sklearn.metrics import mean_absolute_error

    this_guess = pd.Timestamp(this_guess)

    first_hist_obs = model_dict['df_hist'][
        ['deaths_daily', 'cases_daily', 'hosp_admits', 'hosp_concur']].replace(0, np.nan).first_valid_index()
    this_guess = min(
        max(this_guess, first_hist_obs - pd.Timedelta(days=45)),
        first_hist_obs + pd.Timedelta(days=45) )

    last_guess = this_guess
    rmses = pd.Series(dtype='float64')
    change_in_error = -1
    orig_model_dict = model_dict.copy()

    if exposed_0 == None:
        exposed_0 = max(min(model_dict['df_hist']['cases_tot'].max() / 100, 100), 10)
    if infectious_0 == None:
        infectious_0 = max(min(model_dict['df_hist']['cases_tot'].max() / 100, 100), 10)
    # print(infectious_0, exposed_0)

    # Change in error used to be < 0, but this makes a req for a big enough change.
    while ( ( (change_in_error <= 0) or override_cie )
            and ( this_guess >= ( first_hist_obs - pd.Timedelta(days=60) ) )
            and ( this_guess <= ( first_hist_obs + pd.Timedelta(days=60) ) )
    ):
        model_dict = orig_model_dict.copy()
        print('This guess: ', this_guess)
        model_dict['d_to_forecast'] = (pd.Timestamp.today() - this_guess).days
        model_dict = seir_model_cohort(this_guess, model_dict, exposed_0, infectious_0)
        df_agg = model_dict['df_agg']

        # df_compare = pd.DataFrame()
        #
        # if 'deaths_tot' in model_dict['df_hist'].columns:
        #     df_compare['obs_metric'] = model_dict['df_hist']['deaths_tot']
        #     df_compare['pred_metric'] = df_agg['deaths']
        # elif 'hosp_concur' in model_dict['df_hist'].columns:
        #     df_compare['obs_metric'] = model_dict['df_hist']['hosp_concur']
        #     df_compare['pred_metric'] = df_agg['hospitalized']
        #
        # df_compare = df_compare.dropna()
        # df_compare = df_compare.iloc[-60:]
        # rmses.loc[this_guess] = np.sqrt(mean_squared_error(df_compare['obs_metric'], df_compare['pred_metric']))

        df_thiserror = pd.DataFrame()
        if 'deaths_tot' in model_dict['df_hist'].columns:
            df_thiserror['deaths_tot'] = fore_rmse(model_dict['df_hist']['deaths_tot'], df_agg['deaths'])
        if 'deaths_daily' in model_dict['df_hist'].columns:
            df_thiserror['deaths_daily'] = fore_rmse(model_dict['df_hist']['deaths_daily'].rolling(7).mean(), df_agg['deaths_daily'])
        if 'hosp_concur' in model_dict['df_hist'].columns:
            df_thiserror['hosp_concur'] = fore_rmse(model_dict['df_hist']['hosp_concur'], df_agg['hospitalized'])
            df_thiserror.loc['rmse','hosp_concur'] = df_thiserror.loc['rmse','hosp_concur'] * 2
            df_thiserror.loc['avg_error','hosp_concur'] = df_thiserror.loc['avg_error','hosp_concur'] * 2
        rmses.loc[this_guess] = df_thiserror.loc['rmse'].mean()

        print('This rmse: ', rmses.loc[this_guess])

        if last_guess != this_guess:
            change_in_error = rmses.loc[this_guess] - rmses.loc[last_guess]
        print('Change in rmse: ', change_in_error)

        last_guess = this_guess
        # avg_error = df_compare['obs_metric'].sub(df_compare['pred_metric']).mean()
        avg_error = df_thiserror.loc['avg_error'].mean()
        rel_error = df_thiserror.loc['rel_error'].mean()
        print('Average Error: ', avg_error)
        if df_agg['susceptible'].iloc[-1]/model_dict['tot_pop'] < 0.1:
            print('Way too early starting date.')
            this_guess = this_guess + pd.Timedelta(days=14)
            override_cie = True
        elif avg_error < 0:
            if rel_error < 0.5:
                this_guess = this_guess + pd.Timedelta(days=7)
                override_cie = True
            else:
                this_guess = this_guess + pd.Timedelta(days=1)
                override_cie = False
        else:
            if rel_error > 2:
                this_guess = this_guess - pd.Timedelta(days=7)
                override_cie = True
            else:
                this_guess = this_guess - pd.Timedelta(days=1)
                override_cie = False

        # from coronita_chart_helper import ch_deaths_tot, ch_exposed_infectious
        # import matplotlib.pyplot as plt
        # ch_deaths_tot(model_dict); plt.show()
        # ch_exposed_infectious(model_dict);
        # plt.show()

    model_dict = orig_model_dict
    model_dict['d_to_forecast'] = (pd.Timestamp.today() - this_guess).days + model_dict['d_to_forecast']
    model_dict = seir_model_cohort(rmses.idxmin(), model_dict, exposed_0, infectious_0)
    print('Best starting date: ', rmses.idxmin())
    model_dict['rmses'] = rmses
    return model_dict

def est_all_rts(model_dict):
    df_hist = model_dict['df_hist'].copy()
    df_hist = df_hist.dropna(how='all', axis=1)
    df_hist = df_hist[[col for col in df_hist.columns if df_hist[col].std() > 0]]
    df_hist = df_hist.apply(lambda x: x.iloc[:x.to_numpy().nonzero()[0][-1] + 1])

    keepcols = []
    lookback = 7
    d_infect = model_dict['covid_params']['d_infect'] + model_dict['covid_params']['d_incub']
    _gamma = 1 / d_infect

    df_weights = pd.DataFrame(index=df_hist.index)
    df_hist_shifted = pd.DataFrame(index=df_hist.index)
    df_rts_conf = pd.DataFrame()

    if 'cases_daily' in df_hist.columns:
        keepcols.append('cases_daily')
        df_weights['cases_daily'] = 0.5
        cases_shift = int(model_dict['covid_params']['d_incub'] + 2) * -1

        cases_daily = leading_zeros2nan(df_hist['cases_daily']).add(1.0)
        # if outlier_removal(cases_daily, num_std=4).dropna().shape[0] / cases_daily.dropna().shape[0] > 0.8:
        #     cases_daily = outlier_removal(cases_daily, num_std=4)
        # cases_daily = outlier_removal(cases_daily, num_std=4).add(1.0)
        cases_daily = cases_daily.rolling(lookback, center=False, min_periods=1).mean()
        df_hist_shifted['cases_daily'] = cases_daily.shift(cases_shift)

        df_rt = est_rt_wconf(df_hist_shifted['cases_daily'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if ('cases_daily' in df_hist.columns) and ('pos_neg_tests_daily' in df_hist.columns):
        keepcols.append('test_share')
        df_weights['test_share'] = 1.0
        test_share_shift = int(model_dict['covid_params']['d_incub'] + 2) * -1

        cases_daily_7da = leading_zeros2nan(df_hist['cases_daily']).add(1.0)
        # if outlier_removal(cases_daily_7da, num_std=4).dropna().shape[0] / cases_daily_7da.dropna().shape[0] > 0.8:
        #     cases_daily_7da = outlier_removal(cases_daily_7da, num_std=4)
        # cases_daily_7da = outlier_removal(cases_daily_7da).add(1.0)
        cases_daily_7da = cases_daily_7da.rolling(lookback, center=False, min_periods=1).mean()

        pos_neg_tests_7da = model_dict['df_hist']['pos_neg_tests_daily'].add(1.0)
        pos_neg_tests_7da = pos_neg_tests_7da.rolling(lookback, center=False, min_periods=1).mean()
        # if outlier_removal(pos_neg_tests_7da, num_std=4).dropna().shape[0] / pos_neg_tests_7da.dropna().shape[0] > 0.8:
        #     pos_neg_tests_7da = outlier_removal(pos_neg_tests_7da)
        # pos_neg_tests_7da = pos_neg_tests_7da.rolling(lookback, center=False, min_periods=1).mean()

        test_share = cases_daily_7da.div(pos_neg_tests_7da)

        test_share = test_share.replace([np.inf, -np.inf], np.nan)
        test_share = test_share.mask(test_share >= 1)
        test_share = test_share.mask(test_share >= 0.4)
        test_share = test_share.clip(upper=1.0, lower=0.0)
        df_hist_shifted['test_share'] = test_share.shift(test_share_shift)

        df_rt = est_rt_wconf(df_hist_shifted['test_share'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'deaths_daily' in df_hist.columns:
        keepcols.append('deaths_daily')
        df_weights['deaths_daily'] = 2.0
        deaths_shift = int(model_dict['covid_params']['d_incub'] + model_dict['covid_params']['d_til_death']) * -1

        deaths_daily = leading_zeros2nan(df_hist['deaths_daily']).add(1.0)
        deaths_daily = deaths_daily.rolling(lookback, center=False, min_periods=1).mean()
        # if outlier_removal(deaths_daily, num_std=4).dropna().shape[0] / deaths_daily.dropna().shape[0] > 0.8:
        #     deaths_daily = outlier_removal(deaths_daily, num_std=4)
        # else:
        #     print('didnt remove death outliers.')
        # deaths_daily = outlier_removal(deaths_daily, num_std=4).add(1.0)
        # deaths_daily = deaths_daily.rolling(lookback, center=False, min_periods=1).mean()
        df_hist_shifted['deaths_daily'] = deaths_daily.shift(deaths_shift)
        print('deaths shifted by: ', deaths_shift)

        df_rt = est_rt_wconf(df_hist_shifted['deaths_daily'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'hosp_concur' in df_hist.columns:
        keepcols.append('hosp_concur')
        df_weights['hosp_concur'] = 2.0 #1.5
        hosp_concur_shift = (int(model_dict['covid_params']['d_incub']
                                 + model_dict['covid_params']['d_to_hosp']
                                 + model_dict['covid_params']['d_in_hosp'] / 2)
                             * -1)

        hosp_concur = leading_zeros2nan(df_hist['hosp_concur']).add(1.0)
        hosp_concur = hosp_concur.rolling(lookback, center=False, min_periods=1).mean()
        # if outlier_removal(hosp_concur, num_std=4).dropna().shape[0] / hosp_concur.dropna().shape[0] > 0.8:
        #     hosp_concur = outlier_removal(hosp_concur, num_std=4)
        # hosp_concur = outlier_removal(hosp_concur, num_std=4).add(1.0)
        # hosp_concur = hosp_concur.rolling(lookback, center=False, min_periods=1).mean()
        df_hist_shifted['hosp_concur'] = hosp_concur.shift(hosp_concur_shift)

        df_rt = est_rt_wconf(df_hist_shifted['hosp_concur'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'hosp_admits' in df_hist.columns:
        keepcols.append('hosp_admits')
        df_weights['hosp_admits'] = 1.5 #3.0
        hosp_admits_shift = (int(model_dict['covid_params']['d_incub']
                                 + model_dict['covid_params']['d_to_hosp'])
                             * -1)

        if 'hosp_concur' in df_hist.columns:
            df_hist.loc[df_hist['hosp_admits'] < df_hist['hosp_concur'].diff(), 'hosp_admits'] = np.nan

        hosp_admits = leading_zeros2nan(df_hist['hosp_admits'])
        first_hosp_admit = hosp_admits.replace(0, np.nan).dropna().first_valid_index()
        if hosp_admits.loc[first_hosp_admit] > hosp_admits.loc[first_hosp_admit:].quantile(.95):
            first_hosp_admit = first_hosp_admit + pd.Timedelta(days=1)
        hosp_admits = hosp_admits.loc[first_hosp_admit:].reindex(df_hist.index).add(1.0)
        hosp_admits = hosp_admits.rolling(lookback, center=False, min_periods=1).mean()
        # if outlier_removal(hosp_admits, num_std=4).dropna().shape[0] / hosp_admits.dropna().shape[0] > 0.8:
        #     hosp_admits = outlier_removal(hosp_admits, num_std=4)
        # hosp_admits = outlier_removal(hosp_admits, num_std=4).add(1.0)
        # hosp_admits = hosp_admits.rolling(lookback, center=False, min_periods=1).mean()
        df_hist_shifted['hosp_admits'] = hosp_admits.shift(hosp_admits_shift)

        df_rt = est_rt_wconf(df_hist_shifted['hosp_admits'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    df_hist_shifted_ravg = df_hist_shifted.rolling(lookback, win_type='gaussian', center=True, min_periods=3).mean(
        std=1) # Normal
    # df_hist_shifted_ravg = df_hist_shifted.rolling(lookback, center=True, min_periods=3).mean() # Exp. Version for NYS

    df_lambdas = df_hist_shifted_ravg.copy()
    df_lambdas = df_lambdas.pct_change(fill_method=None)
    # df_lambdas = df_lambdas.apply(np.log).diff()
    if 'hosp_concur' in df_lambdas.columns:
        df_lambdas['hosp_concur'] = df_lambdas['hosp_concur'].pow(1.4)
    df_lambdas = df_lambdas.replace([np.inf, -np.inf], np.nan)
    df_lambdas = df_lambdas.apply(outlier_removal, num_std=3)

    df_lambdas = df_lambdas.rolling(lookback, win_type='gaussian', center=True).mean(std=4) # Normal
    # df_lambdas = df_lambdas.rolling(lookback, win_type='gaussian', center=True).mean(std=1) # Exp. Version for NYS

    wavg_lookback = lookback # Normal
    # wavg_lookback = 1 # Exp. Version for NYS
    for center in df_lambdas.dropna(how='all').index:
        bow = center - pd.Timedelta(days=(wavg_lookback - 1) // 2)
        eow = center + pd.Timedelta(days=(wavg_lookback - 1) // 2)
        windowed = np.array(df_lambdas.loc[bow:eow, keepcols].to_numpy()).flatten()
        windowed = windowed[~np.isnan(windowed)]
        weights = df_weights[~df_lambdas[keepcols].isnull()].loc[bow:eow, keepcols].to_numpy().flatten()
        weights = weights[~np.isnan(weights)]

        if windowed.shape[0] > 0:
            weighted_stats = DescrStatsW(windowed, weights=weights, ddof=0)
            df_lambdas.loc[center, 'weighted_average'] = weighted_stats.mean

    ###### Detrended Std Deviation ######
    std_lookback = lookback * 2
    df_detrended = df_lambdas[keepcols].sub(df_lambdas['weighted_average'], axis=0)
    df_detrended = df_detrended.rolling(std_lookback, win_type='gaussian', center=True).mean(std=2)
    df_detrended_abs = df_detrended.apply(np.abs)
    df_detrended_abs_ffill = df_detrended_abs.apply(
        lambda x:
        x.rolling(std_lookback).mean().loc[x.last_valid_index():].fillna(method='ffill') \
            .div(len(df_detrended.columns) * std_lookback).add(1).cumprod().sub(1).add(
            x.dropna().iloc[-1 * std_lookback:].mean())
    )
    df_detrended_abs_bfill = df_detrended_abs.apply(
        lambda x:
        x.rolling(std_lookback).mean().fillna(method='bfill').loc[:x.first_valid_index()][::-1] \
            .div(len(df_detrended.columns) * std_lookback).add(1).cumprod().sub(1).add(
            x.dropna().iloc[:std_lookback].mean())
    )
    df_detrended_abs = df_detrended_abs.fillna(df_detrended_abs_ffill).fillna(df_detrended_abs_bfill)
    # df_detrended_abs = df_detrended_abs.interpolate()

    stddev = df_detrended_abs.apply(np.square).rolling(std_lookback, center=True, min_periods=1).sum().sum(axis=1).div(
        df_detrended_abs.count(axis=1).rolling(std_lookback, center=True, min_periods=1).sum().sub(1)).apply(np.sqrt)

    stddev = stddev.rolling(std_lookback, win_type='gaussian', center=True, min_periods=1).mean(std=2)

    stddev = stddev.fillna(method='ffill').fillna(method='bfill')
    #####################################

    s_lambda = df_lambdas['weighted_average']
    # s_lambda = s_lambda.rolling(lookback, win_type='gaussian', center=False).mean(std=2)
    s_lambda = s_lambda.rolling(lookback, win_type='gaussian', center=True).mean(std=2) # Normal Version
    # s_lambda = s_lambda.rolling(lookback, win_type='gaussian', center=True).mean(std=1) # Exp. Version for NYS

    s_lambda = s_lambda.loc[:s_lambda.last_valid_index()]
    s_lambda = s_lambda.clip(lower=-1.0)

    s_beta = s_lambda.add(_gamma).clip(lower=0.0)

    df_rt = pd.DataFrame(s_beta.div(_gamma))
    df_rt.columns = ['rt']

    df_rt['rt_u68'] = s_lambda.add(stddev.mul(1)).add(_gamma).clip(lower=0.0).div(_gamma)
    df_rt['rt_l68'] = s_lambda.sub(stddev.mul(1)).clip(lower=-1.0).add(_gamma).clip(lower=0).div(_gamma)
    df_rt['rt_u95'] = s_lambda.add(stddev.mul(1.96)).add(_gamma).clip(lower=0.0).div(_gamma)
    df_rt['rt_l95'] = s_lambda.sub(stddev.mul(1.96)).clip(lower=-1.0).add(_gamma).clip(lower=0).div(_gamma)
    df_rt.columns.name = 'metric'
    df_rt = pd.DataFrame(df_rt.stack(), columns=['weighted_average']).unstack()

    df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    model_dict['df_rts_conf'] = df_rts_conf.copy()
    model_dict['df_hist_shifted'] = df_hist_shifted.copy()
    model_dict['df_lambdas'] = df_lambdas.copy()


    # cases_daily.dropna().plot(figsize=[14, 4]);
    # plt.show()
    # df_rts_conf.unstack('metric').swaplevel(axis=1)['rt'].plot(figsize=[14, 4], title='rts');
    # plt.show()
    # df_rts_conf.unstack('metric').swaplevel(axis=1)['rt']['weighted_average'].plot(figsize=[14, 4]);
    # plt.show()
    # df_lambdas.plot(figsize=[14, 4], title='lambdas');
    # plt.show()
    # df_detrended.plot(title='Detrended', figsize=[14, 4]);
    # plt.show()
    # df_detrended_abs.plot(title='Detrended', figsize=[14, 4]);
    # plt.show()
    # stddev.plot(title='stddev', figsize=[14, 4]);
    # plt.show()
    return model_dict

def est_rt_wconf(lvl_series, lookback, d_infect):
    last_nonzero_idx = lvl_series.to_numpy().nonzero()[0][-1]
    lvl_series = lvl_series.iloc[:last_nonzero_idx+1]

    _gamma = 1 / d_infect

    pct_series = lvl_series.rolling(lookback, center=True).mean().pct_change(fill_method=None)
    pct_series = pct_series.replace([np.inf, -np.inf], np.nan)

    stddev = pct_series.rolling(lookback,
                                             win_type='gaussian',
                                             min_periods=lookback-1,
                                             center=True
                                             ).std(std=2).fillna(method='ffill').fillna(method='bfill')

    s_lambda = pct_series.rolling(lookback,
                                               win_type='gaussian',
                                               min_periods=lookback-1,
                                               center=True
                                               ).mean(std=2)

    s_lambda = s_lambda.loc[:s_lambda.last_valid_index()]
    s_lambda = s_lambda.clip(lower=-1.0)

    s_beta = s_lambda.add(_gamma).clip(lower=0.0)

    df_rt = pd.DataFrame(s_beta.div(_gamma))
    df_rt.columns = ['rt']

    df_rt['rt_u68'] = s_lambda.add(stddev.mul(1)).add(_gamma).clip(lower=0.0).div(_gamma)
    df_rt['rt_l68'] = s_lambda.sub(stddev.mul(1)).clip(lower=-1.0).add(_gamma).clip(lower=0).div(_gamma)
    df_rt['rt_u95'] = s_lambda.add(stddev.mul(1.96)).add(_gamma).clip(lower=0.0).div(_gamma)
    df_rt['rt_l95'] = s_lambda.sub(stddev.mul(1.96)).clip(lower=-1.0).add(_gamma).clip(lower=0).div(_gamma)
    df_rt.columns.name = 'metric'
    df_rt = pd.DataFrame(df_rt.stack(), columns=[lvl_series.name]).unstack()

    return df_rt

def make_model_dict_state(state_code, abbrev_us_state, df_census, df_st_testing_fmt, df_hhs_hosp, df_can, df_counties, df_vax_hes,
                          covid_params, d_to_forecast = 75, df_mvmt=pd.DataFrame(), df_interventions=pd.DataFrame()):
    model_dict = {}

    model_dict['region_code'] = state_code
    model_dict['region_name'] = abbrev_us_state[model_dict['region_code']]
    model_dict['tot_pop'] = df_census.loc[(df_census.SUMLEV == 40)
                                                  & (df_census.state == model_dict['region_code']),
                                                  'pop2019'].values[0]
    model_dict['tot_pop_18plus'] = df_census.loc[(df_census.SUMLEV == 40)
                                                  & (df_census.state == model_dict['region_code']),
                                                  'pop2019_18plus'].values[0]

    model_dict['df_hist'] = pd.DataFrame()

    if df_st_testing_fmt['deaths'][model_dict['region_code']].dropna().shape[0] > 0:
        model_dict['df_hist']['deaths_tot'] = df_st_testing_fmt['deaths'][model_dict['region_code']]
        deaths_daily = model_dict['df_hist']['deaths_tot'].diff()
        model_dict['df_hist']['deaths_daily'] = deaths_daily.mask(deaths_daily < 0)

    if df_st_testing_fmt['hospitalizedCurrently'][model_dict['region_code']].dropna().shape[0] > 0:
        model_dict['df_hist']['hosp_concur'] = df_st_testing_fmt['hospitalizedCurrently'][model_dict['region_code']]

    if df_st_testing_fmt['hospitalizedIncrease'][model_dict['region_code']].dropna().shape[0] > 0:
        hosp_admits = df_st_testing_fmt['hospitalizedIncrease'][model_dict['region_code']]
        model_dict['df_hist']['hosp_admits'] = hosp_admits.mask(hosp_admits < 0)

    if state_code == 'NY':
        model_dict['df_hist'].loc['2020-06-04': ,'hosp_admits'] = hosp_admits.mask(hosp_admits == 0)

    model_dict['df_hist']['cases_tot'] = df_st_testing_fmt['cases'][model_dict['region_code']]
    cases_daily = model_dict['df_hist']['cases_tot'].diff()
    model_dict['df_hist']['cases_daily'] = cases_daily.mask(cases_daily < 0)

    model_dict['df_hist']['pos_neg_tests_tot'] = df_st_testing_fmt['posNeg'][model_dict['region_code']]
    pos_neg_tests_daily = model_dict['df_hist']['pos_neg_tests_tot'].diff()
    model_dict['df_hist']['pos_neg_tests_daily'] = pos_neg_tests_daily.mask(pos_neg_tests_daily < 0)

    try:
        model_dict['df_hist']['hosp_beds_tot'] = df_hhs_hosp['Total Inpatient Beds'].loc[model_dict['region_code']]
        model_dict['df_hist']['hosp_beds_avail'] = df_hhs_hosp['hosp_beds_avail'].loc[model_dict['region_code']]
    except:
        print('hosp capacity data not available for {}'.format(model_dict['region_code']))

    model_dict['covid_params'] = covid_params.copy()

    if model_dict['df_hist']['deaths_daily'].mean() > 0.5:
        # deaths_hosp_rat = np.linalg.lstsq(model_dict['df_hist'][['deaths_daily','hosp_concur']].dropna()['deaths_daily'].values.reshape(-1, 1),
        #                                   model_dict['df_hist'][['deaths_daily','hosp_concur']].dropna()['hosp_concur'],
        #                                   rcond=None)[0][0] / 9.0
        deaths_shift = int(model_dict['covid_params']['d_incub'] + model_dict['covid_params']['d_til_death']) * -1
        hosp_concur_shift = (int(model_dict['covid_params']['d_incub']
                                 + model_dict['covid_params']['d_to_hosp']
                                 + model_dict['covid_params']['d_in_hosp'] / 2)
                             * -1)

        df_dhratio = model_dict['df_hist'][['deaths_daily', 'hosp_concur']].copy()

        df_dhratio['deaths_daily'] = df_dhratio['deaths_daily'].rolling(7).sum().shift(deaths_shift)
        df_dhratio['hosp_concur'] = df_dhratio['hosp_concur'].rolling(7).mean().shift(hosp_concur_shift)
        df_dhratio = df_dhratio.dropna()

        deaths_hosp_rat = np.linalg.lstsq(df_dhratio['deaths_daily'].values.reshape(-1, 1),
                                          df_dhratio['hosp_concur'], rcond=None)[0][0]

        old_ratio = model_dict['covid_params']['hosp_rt'] / model_dict['covid_params']['mort_rt']
        model_dict['covid_params']['hosp_rt'] = ((deaths_hosp_rat / old_ratio - 1) / 2 + 1) * model_dict['covid_params']['hosp_rt']
        model_dict['covid_params']['mort_rt'] = model_dict['covid_params']['hosp_rt'] / deaths_hosp_rat

    model_dict = est_all_rts(model_dict)
    model_dict['df_rts'] = model_dict['df_rts_conf'].unstack().swaplevel(axis=1)['rt']
    model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['weighted_average'].max()

    model_dict['d_to_forecast'] = int(d_to_forecast)

    if df_mvmt.shape[0] > 0:
        model_dict['df_mvmt'] = df_mvmt.loc[state_code]
    else:
        model_dict['df_mvmt'] = df_mvmt

    if df_interventions.shape[0] > 0:
        model_dict['df_interventions'] = df_interventions[
            df_interventions.state_code.isin([state_code,'US'])].groupby('dt').first().reset_index()
    else:
        model_dict['df_interventions'] = df_interventions

    model_dict['df_hist']['vax_initiated'] = df_can.loc[model_dict['region_code'], 'actuals.vaccinationsInitiated'].replace(0, np.nan)
    model_dict['df_hist']['vax_completed'] = df_can.loc[model_dict['region_code'], 'actuals.vaccinationsCompleted'].replace(0, np.nan)
    model_dict['df_hist']['vax_halfcompleted'] = model_dict['df_hist']['vax_initiated'] - model_dict['df_hist']['vax_completed']

    model_dict['df_vax_hes'] = df_vax_hes.query(f"state=='{state_code}'")
    df_vax_hes_state = calc_vax_hes_state(df_vax_hes)
    model_dict['covid_params']['est_vax_hes_pop_18plus'] = df_vax_hes_state.loc[state_code, 'est_vax_hes_pop_18plus']
    model_dict['covid_params']['est_vax_hes_strong_pop_18plus'] = df_vax_hes_state.loc[state_code, 'est_vax_hes_strong_pop_18plus']
    model_dict['df_vax_fore'] = forecast_vaccines(model_dict)

    model_dict['df_counties'] = df_counties.query(f"state=='{state_code}'")[['pop2019', 'cases', 'deaths', 'cases_per100k']]

    model_dict['footnote_str'] = ''
    model_dict['chart_title'] = ''

    return model_dict

def make_model_dict_us(df_census, df_st_testing_fmt, df_hhs_hosp, df_can, df_counties, df_vax_hes, covid_params, d_to_forecast = 75,
                        df_mvmt=pd.DataFrame(), df_interventions=pd.DataFrame()):
    model_dict = {}

    model_dict['region_code'] = 'US'
    model_dict['region_name'] = 'United States'
    model_dict['tot_pop'] = df_census.loc[(df_census.SUMLEV == 40), 'pop2019'].sum()
    model_dict['tot_pop_18plus'] = df_census.loc[(df_census.SUMLEV == 40), 'pop2019_18plus'].sum()

    model_dict['df_hist'] = pd.DataFrame()

    model_dict['df_hist']['deaths_tot'] = df_st_testing_fmt['deaths'][df_census.state.unique()].sum(axis=1, skipna=False).dropna()
    model_dict['df_hist']['deaths_daily'] = model_dict['df_hist']['deaths_tot'].diff()

    model_dict['df_hist']['cases_tot'] = df_st_testing_fmt['cases'][df_census.state.unique()].sum(axis=1, skipna=False).dropna()
    model_dict['df_hist']['cases_daily'] = model_dict['df_hist']['cases_tot'].diff()
    model_dict['df_hist']['pos_neg_tests_tot'] = df_st_testing_fmt['posNeg'][df_census.state.unique()].sum(axis=1, skipna=False).dropna()
    model_dict['df_hist']['pos_neg_tests_daily'] = model_dict['df_hist']['pos_neg_tests_tot'].diff()

    try:
        model_dict['df_hist']['hosp_beds_tot'] = df_hhs_hosp['Total Inpatient Beds'].loc[model_dict['region_code']]
        model_dict['df_hist']['hosp_beds_avail'] = df_hhs_hosp['hosp_beds_avail'].loc[model_dict['region_code']]
    except:
        print('hosp capacity data not available for {}'.format(model_dict['region_code']))

    model_dict['covid_params'] = covid_params

    # model_dict['df_rts_conf'] = est_all_rts(model_dict)
    model_dict = est_all_rts(model_dict)
    model_dict['df_rts'] = model_dict['df_rts_conf'].unstack().swaplevel(axis=1)['rt']
    model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['weighted_average'].max()
    model_dict['covid_params']['current_r0'] = min(max(model_dict['covid_params']['basic_r0'] * 1.2, 2.0), 3.0)

    model_dict['d_to_forecast'] = int(d_to_forecast)

    if df_mvmt.shape[0] > 0:
        model_dict['df_mvmt'] = df_mvmt[df_mvmt.state.isnull()].set_index('dt')
    else:
        model_dict['df_mvmt'] = df_mvmt

    if df_interventions.shape[0] > 0:
        model_dict['df_interventions'] = df_interventions[
            df_interventions.state_code.isin(['US'])].groupby('dt').first().reset_index()
    else:
        model_dict['df_interventions'] = df_interventions

    model_dict['df_hist']['vax_initiated'] = df_can['actuals.vaccinationsInitiated'].unstack('state').replace(0, np.nan).dropna(thresh=40).sum(axis=1)
    model_dict['df_hist']['vax_completed'] = df_can['actuals.vaccinationsCompleted'].unstack('state').replace(0, np.nan).dropna(thresh=40).sum(axis=1)
    model_dict['df_hist']['vax_halfcompleted'] = model_dict['df_hist']['vax_initiated'] - model_dict['df_hist'][
        'vax_completed']

    model_dict['df_vax_hes'] = df_vax_hes
    df_vax_hes_state = df_vax_hes.sum()[
        ['pop2019', 'pop2019_18plus', 'est_vax_hes_pop_18plus', 'est_vax_hes_strong_pop_18plus']]
    model_dict['covid_params']['est_vax_hes_pop_18plus'] = df_vax_hes_state['est_vax_hes_pop_18plus']
    model_dict['covid_params']['est_vax_hes_strong_pop_18plus'] = df_vax_hes_state['est_vax_hes_strong_pop_18plus']
    model_dict['df_vax_fore'] = forecast_vaccines(model_dict)


    model_dict['df_counties'] = df_counties[['pop2019', 'cases', 'deaths', 'cases_per100k']]

    model_dict['footnote_str'] = ''
    model_dict['chart_title'] = ''

    return model_dict


def calc_vax_hes_state(df_vax_hes):
    df_vax_hes_state = df_vax_hes.groupby('state').sum()[['pop2019', 'pop2019_18plus', 'est_vax_hes_pop_18plus', 'est_vax_hes_strong_pop_18plus']]
    df_vax_hes_state['est_vax_hes'] = df_vax_hes_state['est_vax_hes_pop_18plus'] / df_vax_hes_state['pop2019_18plus']
    df_vax_hes_state['est_vax_hes_strong'] = df_vax_hes_state['est_vax_hes_strong_pop_18plus'] / df_vax_hes_state['pop2019_18plus']
    return df_vax_hes_state

def forecast_vaccines(model_dict):
    df = model_dict['df_hist'][['vax_initiated']].reset_index()
    df = df.rename(columns={'dt': 'ds', 'vax_initiated': 'y'})
    df['cap'] = model_dict['tot_pop_18plus'] - model_dict['covid_params']['est_vax_hes_pop_18plus']
    m = Prophet(growth='logistic', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=365*2)
    future['cap'] = model_dict['tot_pop_18plus'] - model_dict['covid_params']['est_vax_hes_pop_18plus']
    fcst = m.predict(future)
    fcst = fcst.set_index('ds')[['trend', 'cap', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper']]
    return fcst

def leading_zeros2nan(input_series):
    first_nonzero_idx = input_series[input_series > 0].first_valid_index()
    input_series.loc[:first_nonzero_idx] = np.nan
    return input_series