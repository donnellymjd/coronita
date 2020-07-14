from scipy.stats import gamma
from statsmodels.stats.weightstats import DescrStatsW
from coronita_chart_helper import *

def outlier_removal(raw_series, num_std=3):
    rolling_avg = raw_series.rolling(7, center=True, min_periods=3).mean().fillna(method='bfill').fillna(method='ffill')
    noise = raw_series - rolling_avg
    cleaned_series = raw_series[(np.abs(noise) < (noise.std() * num_std))].reindex(raw_series.index)
    return cleaned_series

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
    suspop = [model_dict['tot_pop'] - exposed_0 - infectious_0]
    next_infectious = infectious_0
    next_hospitalized = 0
    _gamma = 1 / (model_dict['covid_params']['d_infect'])

    t = np.linspace(0, model_dict['d_to_forecast'], model_dict['d_to_forecast'] + 1)

    df_all_cohorts = pd.DataFrame()
    df_all_cohorts.columns.name = 'cohort_dt'

    r_t = pd.Series(np.nan, index=pd.date_range(start_dt,
                                                start_dt + pd.Timedelta(days=model_dict['d_to_forecast']) ) )
    if 'rt_scenario' in model_dict['df_rts'].columns:
        r_t = r_t.fillna(model_dict['df_rts']['rt_scenario']).fillna(method='bfill').fillna(method='ffill')
    else:
        r_t = r_t.fillna(model_dict['df_rts']['weighted_average']).fillna(method='bfill').fillna(method='ffill')

    model_dict['df_rts'] = model_dict['df_rts'].reindex(r_t.index)
    model_dict['df_rts']['policy_triggered'] = 0

    last_r = r_t.iloc[0]

    for t_ in t[:-1]:
        cohort_strt = start_dt + pd.Timedelta(days=t_)

        if (model_dict['covid_params']['policy_trigger']
                and (cohort_strt > model_dict['df_rts']['weighted_average'].last_valid_index()) ):
            tot_hosp_capacity = model_dict['tot_pop']/1000 * 2.7
            covid_hosp_capacity = tot_hosp_capacity*0.2
            if ( (next_hospitalized > covid_hosp_capacity)
                    or (model_dict['covid_params']['policy_trigger_once']
                        and model_dict['df_rts']['policy_triggered'].sum() > 1) ):
                r_t.loc[cohort_strt] = 0.9
                model_dict['df_rts'].loc[cohort_strt, 'policy_triggered'] = 1

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
            dS = -1 * min(beta * suspop[-1] * next_infectious / model_dict['tot_pop'], suspop[-1])

            df_daily_cohort = dS * -1 * df_daily_cohort_scalar.iloc[:int(d_to_fore)] / 1e6

            df_daily_cohort.index = pd.date_range(cohort_strt,
                                                  cohort_strt + pd.Timedelta(days=d_to_fore - 1))
            df_all_cohorts[cohort_strt] = df_daily_cohort.stack()
            # print(cohort_strt, ' rt: ', this_r)

        d_cohort_totpop_std = round(df_daily_cohort[
                                        ['exposed', 'deaths', 'hospitalized', 'infectious', 'recovered']
                                    ].dropna().sum(axis=1).std(), 3)

        if d_cohort_totpop_std != 0.0:
            print(cohort_strt, d_cohort_totpop_std)
            display(df_daily_cohort)
            raise Exception('Daily Cohort total population varies significantly')

        df_agg = df_all_cohorts.sum(axis=1).unstack()
        df_agg.index = pd.DatetimeIndex(df_agg.index).normalize()
        next_infectious = df_agg.loc[cohort_strt, 'infectious']
        next_hospitalized = df_agg.loc[cohort_strt, 'hospitalized']
        next_suspop = suspop[-1] + dS
        suspop.append(next_suspop)

        totpopchk = df_agg.loc[cohort_strt, ['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths']].sum()

        if (round(totpopchk + suspop[-1]) != round(model_dict['tot_pop'])):
            display(df_all_cohorts)
            display(df_all_cohorts.sum(axis=1).unstack())
            print(cohort_strt)
            print('totpop: ', round(model_dict['tot_pop']))
            print('dS ', dS)
            print('sum of df_agg', totpopchk)
            print('suspop[-1]', suspop[-1])
            print('sum of both', round(totpopchk + suspop[-1]))
            raise Exception('Agg total population varies significantly')

    model_dict['df_rts']['rt_scenario'] = r_t

    df_agg = df_all_cohorts.sum(axis=1).unstack()
    df_agg.index = pd.DatetimeIndex(df_agg.index).normalize()

    s_suspop = pd.Series(suspop, index=pd.date_range(
        start_dt - pd.Timedelta(days=1),
        start_dt + pd.Timedelta(days=model_dict['d_to_forecast'] - 1)))

    df_agg['susceptible'] = pd.Series(s_suspop)
    exposed_daily = df_all_cohorts.stack().unstack(['metric'])[['exposed']].reset_index()
    df_agg['exposed_daily'] = exposed_daily[(exposed_daily.dt == exposed_daily.cohort_dt)].set_index(['dt'])['exposed']

    return df_agg.dropna(), df_all_cohorts, model_dict

def model_find_start(this_guess, model_dict, exposed_0=None, infectious_0=None):
    from sklearn.metrics import mean_squared_error
    # from sklearn.metrics import mean_absolute_error

    last_guess = this_guess
    rmses = pd.Series(dtype='float64')
    change_in_error = -1
    orig_model_dict = model_dict.copy()

    if exposed_0 == None:
        exposed_0 = max(min(model_dict['df_hist']['cases_tot'].max() / 100, 100), 10)
    if infectious_0 == None:
        infectious_0 = max(min(model_dict['df_hist']['cases_tot'].max() / 100, 100), 10)
    # print(infectious_0, exposed_0)

    while change_in_error < 0:
        # print(this_guess)
        model_dict['d_to_forecast'] = (pd.Timestamp.today() - this_guess).days
        df_agg, df_all_cohorts, _ = seir_model_cohort(this_guess, model_dict, exposed_0, infectious_0)

        df_compare = pd.DataFrame()

        if 'deaths_tot' in model_dict['df_hist'].columns:
            df_compare['obs_metric'] = model_dict['df_hist']['deaths_tot']
            df_compare['pred_metric'] = df_agg['deaths']
        elif 'hosp_concur' in model_dict['df_hist'].columns:
            df_compare['obs_metric'] = model_dict['df_hist']['hosp_concur']
            df_compare['pred_metric'] = df_agg['hospitalized']

        df_compare = df_compare.dropna()
        rmses.loc[this_guess] = np.sqrt(mean_squared_error(df_compare['obs_metric'], df_compare['pred_metric']))

        # print(rmses.loc[this_guess])

        if last_guess != this_guess:
            change_in_error = rmses.loc[this_guess] - rmses.loc[last_guess]
        # print(change_in_error)

        last_guess = this_guess
        avg_error = df_compare['obs_metric'].sub(df_compare['pred_metric']).mean()
        # print(avg_error)
        if avg_error < 0:
            this_guess = this_guess + pd.Timedelta(days=1)
        else:
            this_guess = this_guess - pd.Timedelta(days=1)

    model_dict = orig_model_dict
    model_dict['d_to_forecast'] = (pd.Timestamp.today() - this_guess).days + model_dict['d_to_forecast']
    df_agg, df_all_cohorts, model_dict = seir_model_cohort(rmses.idxmin(), model_dict, exposed_0, infectious_0)
    print('Best starting date: ',rmses.idxmin())
    return df_agg, df_all_cohorts, model_dict

def est_rt(df_input, lookback, d_infect, offset_days):
    _gamma = 1 / d_infect
#     df_beta = df_input.pct_change().rolling(lookback,center=True).mean().add(_gamma)
<<<<<<< HEAD
    s_lambda = df_input.rolling(lookback,
                               win_type='gaussian',
                               min_periods=lookback//2,
                               center=True
                               ).mean(std=2).pct_change()
    s_lambda = s_lambda.loc[:s_lambda.last_valid_index()]
    s_beta = s_lambda.rolling(lookback,
                               win_type='gaussian',
                               min_periods=lookback,
                               center=True
                               ).mean(std=2).add(_gamma)
    # s_beta = s_lambda.add(_gamma)

    s_rt = s_beta.div(_gamma)
    s_rt.index = (s_rt.index - pd.Timedelta(days=(offset_days-2))).normalize()
    s_rt = s_rt.dropna().clip(lower=0.0)
    return s_rt
=======
    _lambda = df_input.rolling(lookback,
                               win_type='gaussian',
                               min_periods=lookback,
                               center=False
                               ).mean(std=2).pct_change()
    _lambda = _lambda.loc[:_lambda.last_valid_index()]
    _beta = _lambda.rolling(lookback, center=False).mean().add(_gamma)
    # _beta = _lambda.add(_gamma)

    df_rt = _beta.div(_gamma)
    df_rt.index = (df_rt.index - pd.Timedelta(days=(offset_days + (lookback-2)))).normalize()
    df_rt = df_rt.dropna().clip(lower=0.0)
    return df_rt
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa

def est_all_rts(model_dict):
    df_rts = pd.DataFrame(index=model_dict['df_hist']['cases_daily'].index)

    if 'deaths_daily' in model_dict['df_hist'].columns:
        deaths_daily = model_dict['df_hist']['deaths_daily']
        # deaths_daily = deaths_daily.mask(deaths_daily.rolling(14, win_type='gaussian', center=False).mean(std=2) < 1)
        df_rts['rt_deaths_daily'] = est_rt(deaths_daily,
                                           14,
                                           model_dict['covid_params']['d_infect'],
                                           model_dict['covid_params']['d_incub'] + model_dict['covid_params'][
                                               'd_til_death']
                                           )

    if 'hosp_concur' in model_dict['df_hist'].columns:
        hosp_concur = model_dict['df_hist']['hosp_concur']
        # hosp_concur = hosp_concur.mask(hosp_concur.rolling(7, win_type='gaussian', center=False).mean(std=2) < 1)
        df_rts['rt_hosp_concur'] = est_rt(hosp_concur,
                                          7,
                                          model_dict['covid_params']['d_infect'],
                                          (model_dict['covid_params']['d_incub']
                                           + model_dict['covid_params']['d_to_hosp']
                                           + model_dict['covid_params']['d_in_hosp'] / 2
                                           ))

    if 'hosp_admits' in model_dict['df_hist'].columns:
        hosp_admits = model_dict['df_hist']['hosp_admits']
        # hosp_admits = hosp_admits.mask(hosp_admits.rolling(7, win_type='gaussian', center=False).mean(std=2) < 1)
        df_rts['rt_hosp_admits'] = est_rt(hosp_admits,
                                          7,
                                          model_dict['covid_params']['d_infect'],
                                          (model_dict['covid_params']['d_incub']
                                           + model_dict['covid_params']['d_to_hosp']
                                           ))

    if 'cases_daily' in model_dict['df_hist'].columns:
        cases_daily = model_dict['df_hist']['cases_daily']
        cases_daily = cases_daily.mask(cases_daily.rolling(7, win_type='gaussian', center=True).mean(std=2) < 1)
        df_rts['rt_cases_daily'] = est_rt(cases_daily,
                                          7,
                                          model_dict['covid_params']['d_infect'],
                                          (model_dict['covid_params']['d_incub']
                                          ))

    if ('cases_daily' in model_dict['df_hist'].columns) and ('pos_neg_tests_daily' in model_dict['df_hist'].columns):
        cases_daily = model_dict['df_hist']['cases_daily']
        cases_daily = cases_daily.mask(cases_daily.rolling(7, win_type='gaussian', center=True).mean(std=2) < 1)
        pos_neg_tests_daily = model_dict['df_hist']['pos_neg_tests_daily']
<<<<<<< HEAD
        pos_neg_tests_daily = pos_neg_tests_daily.mask(pos_neg_tests_daily <= 0)
=======
        pos_neg_tests_daily = pos_neg_tests_daily.mask(pos_neg_tests_daily < 0)
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa
        test_share = cases_daily.div(pos_neg_tests_daily)
        test_share = test_share.mask(test_share >= 1)

        df_rts['rt_pos_test_share_daily'] = est_rt(
            test_share,
            7,
            model_dict['covid_params']['d_infect'],
            (model_dict['covid_params']['d_incub']
            ))

    ## Drop estimates with very high standard deviations ##
    five_stds = df_rts.std().median() * 5
    col_stds = df_rts.std()
    l_donotavg = col_stds[col_stds > five_stds].index.to_list() + col_stds[col_stds.gt(1)].index.to_list()
    # df_rts = df_rts.drop(columns=col_stds[col_stds > five_stds].index)
    # df_rts = df_rts.drop(columns=col_stds[col_stds.gt(1)].index)

    daysatzero = df_rts[df_rts == 0].count()
    df_rts = df_rts.drop(columns=daysatzero[daysatzero > 3].index)

    primary_rts = ['rt_deaths_daily', 'rt_hosp_concur', 'rt_hosp_admits']
    avail_prim_rts = [col for col in df_rts.columns if (col in primary_rts) and (col not in l_donotavg)]
<<<<<<< HEAD
    df_rts['rt_primary_mean'] = df_rts[avail_prim_rts].mean(skipna=True, axis=1)
    df_rts['rt_primary_mean'] = df_rts['rt_primary_mean'].interpolate(limit_area='inside')
    df_rts['rt_primary_mean'] = df_rts['rt_primary_mean'].dropna().rolling(
        7,
        win_type='gaussian',
        min_periods=7,
=======
    df_rts['rt_primary_mean'] = df_rts[avail_prim_rts].mean(axis=1).interpolate(limit_area='inside').rolling(7,
        win_type='gaussian',
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa
        center=True).mean(std=2)

    # secondary_rt = 'rt_pos_test_share_daily' if 'rt_pos_test_share_daily' in df_rts.columns else 'rt_cases_daily'
    # secondary_index = pd.Series(1.0,
    #                             index=pd.date_range(df_rts['rt_primary_mean'].last_valid_index(),
    #                                                 df_rts[secondary_rt].last_valid_index()))
    # secondary_index = secondary_index.add(df_rts[secondary_rt].loc[secondary_index.index].pct_change(),
    #                                       fill_value=0)
    # secondary_index = secondary_index.cumprod()
    # df_rts['rt_secondary_est'] = secondary_index.mul(df_rts['rt_primary_mean'].dropna().iloc[-1])

    secondary_rts = ['rt_pos_test_share_daily', 'rt_cases_daily']
    avail_sec_rts = [col for col in df_rts.columns if (col in secondary_rts) and (col not in l_donotavg)]
<<<<<<< HEAD
    df_rts['rt_secondary_est'] = df_rts[avail_sec_rts].mean(skipna=True, axis=1)
    df_rts['rt_secondary_est'] = df_rts['rt_secondary_est'].interpolate(limit_area='inside')
    df_rts['rt_secondary_est'] = df_rts['rt_secondary_est'].dropna().rolling(
        7,
=======
    df_rts['rt_secondary_est'] = df_rts[avail_sec_rts].mean(axis=1).interpolate(limit_area='inside').rolling(7,
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa
        win_type='gaussian',
        min_periods=7,
        center=True).mean(std=2)

    # df_rts['rt_joint_est'] = df_rts['rt_primary_mean'].fillna(df_rts['rt_secondary_est']).rolling(7,
    #     win_type='gaussian',
    #     min_periods=3,
    #     center=True).mean(std=4)
    df_rts['rt_joint_est'] = df_rts['rt_primary_mean'].fillna(df_rts['rt_secondary_est'])

<<<<<<< HEAD
    # df_rts['rt_joint_est'] = df_rts['rt_primary_mean'].fillna(
    #     df_rts.loc[df_rts['rt_primary_mean'].last_valid_index():, 'rt_secondary_est'])

    max_in_q1 = df_rts['rt_joint_est'].loc['2020-01-01':'2020-03-31'].idxmax()
    df_rts['rt_joint_est'] = df_rts['rt_joint_est'].loc[max_in_q1:]
=======
    # df_rts['rt_joint_est'] = df_rts['rt_joint_est'].loc[df_rts['rt_joint_est'].idxmax():]
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa

    return df_rts

def est_all_rts_new(model_dict):
    df_lambdas = model_dict['df_hist'].copy()

    keepcols = []
    lookback = 7
    d_infect = model_dict['covid_params']['d_infect'] + model_dict['covid_params']['d_incub']
    _gamma = 1 / d_infect

    df_weights = pd.DataFrame(index=df_lambdas.index)
    df_rts_conf = pd.DataFrame()

    if ('cases_daily' in df_lambdas.columns) and ('pos_neg_tests_daily' in df_lambdas.columns):
        keepcols.append('test_share')
        df_weights['test_share'] = 1.0

        df_lambdas['test_share'] = outlier_removal(df_lambdas['cases_daily']).div(
            outlier_removal(df_lambdas['pos_neg_tests_daily']))

        df_lambdas['test_share'] = df_lambdas['test_share'].replace([np.inf, -np.inf], np.nan)
        df_lambdas['test_share'] = df_lambdas['test_share'].mask(df_lambdas['test_share'] >= 1)
        df_lambdas['test_share'] = df_lambdas['test_share'].clip(upper=1.0, lower=0.0)
        df_lambdas['test_share'] = df_lambdas['test_share'].shift(
            int(model_dict['covid_params']['d_incub'])*-1)
        df_rt = est_rt_wconf(df_lambdas['test_share'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'deaths_daily' in df_lambdas.columns:
        keepcols.append('deaths_daily')
        df_weights['deaths_daily'] = 2.0
        df_lambdas['deaths_daily'] = df_lambdas['deaths_daily'].shift(
            int(model_dict['covid_params']['d_incub']
            + model_dict['covid_params']['d_til_death'])*-1)
        df_rt = est_rt_wconf(df_lambdas['deaths_daily'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'hosp_concur' in df_lambdas.columns:
        keepcols.append('hosp_concur')
        df_weights['hosp_concur'] = 4.0
        df_lambdas['hosp_concur'] = df_lambdas['hosp_concur'].shift(
            int(model_dict['covid_params']['d_incub']
                + model_dict['covid_params']['d_to_hosp']
                + model_dict['covid_params']['d_in_hosp'] / 2)*-1)
        df_rt = est_rt_wconf(df_lambdas['hosp_concur'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'hosp_admits' in df_lambdas.columns:
        keepcols.append('hosp_admits')
        df_weights['hosp_admits'] = 4.0

        if 'hosp_concur' in df_lambdas.columns:
            df_lambdas.loc[
                model_dict['df_hist']['hosp_admits'] < model_dict['df_hist']['hosp_concur'].diff(), 'hosp_admits'] = np.nan

        df_lambdas['hosp_admits'] = df_lambdas['hosp_admits'].shift(
            int(model_dict['covid_params']['d_incub'] + model_dict['covid_params']['d_to_hosp'])*-1)
        df_rt = est_rt_wconf(df_lambdas['hosp_admits'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    if 'cases_daily' in df_lambdas.columns:
        keepcols.append('cases_daily')
        df_weights['cases_daily'] = 0.5
        df_lambdas['cases_daily'] = df_lambdas['cases_daily'].shift(
            int(model_dict['covid_params']['d_incub'] + 2)*-1)
        df_rt = est_rt_wconf(df_lambdas['cases_daily'], lookback, d_infect)
        df_rts_conf = pd.concat([df_rts_conf, df_rt.stack('metric')], axis=1)

    df_lambdas = df_lambdas[keepcols]
    df_lambdas = df_lambdas.apply(outlier_removal)
    df_lambdas = df_lambdas.rolling(lookback, min_periods=lookback-1, center=True).mean()
    df_lambdas = df_lambdas.pct_change(fill_method=None)
    df_lambdas = df_lambdas.replace([np.inf, -np.inf], np.nan)
    df_lambdas = df_lambdas.apply(outlier_removal, num_std=3)

    df_lambdas['daily_mean'] = df_lambdas[keepcols].mean(axis=1)

    for center in df_lambdas.dropna(how='all').index:
        bow = center - pd.Timedelta(days=(lookback-1)//2)
        eow = center + pd.Timedelta(days=(lookback-1)//2)
        windowed = np.array(df_lambdas.loc[bow:eow, keepcols].to_numpy()).flatten()
        windowed = windowed[~np.isnan(windowed)]
        weights = df_weights[~df_lambdas[keepcols].isnull()].loc[bow:eow, keepcols].to_numpy().flatten()
        weights = weights[~np.isnan(weights)]

        if windowed.shape[0] > 0:
            weighted_stats = DescrStatsW(windowed, weights=weights, ddof=0)
            df_lambdas.loc[center, 'weighted_average'] = weighted_stats.mean

    ###### Detrended Std Deviation ######
    df_detrended = df_lambdas[keepcols].sub(df_lambdas['weighted_average'], axis=0)
    for center in df_lambdas.dropna(how='all').index:
        bow = center - pd.Timedelta(days=(lookback-1)//2)
        eow = center + pd.Timedelta(days=(lookback-1)//2)
        windowed = np.array(df_detrended.loc[bow:eow, keepcols].to_numpy()).flatten()
        windowed = windowed[~np.isnan(windowed)]
        weights = df_weights[~df_detrended[keepcols].isnull()].loc[bow:eow, keepcols].to_numpy().flatten()
        weights = weights[~np.isnan(weights)]

        if windowed.shape[0] > 0:
            weighted_stats = DescrStatsW(windowed, weights=weights, ddof=0)
            this_std = weighted_stats.std

            if this_std == 0.0:
                this_std = np.nan

            df_lambdas.loc[center, 's_comb_std'] = this_std
    #####################################

    stddev = df_lambdas['s_comb_std'].fillna(method='ffill').fillna(method='bfill')

    s_lambda = df_lambdas['weighted_average'].rolling(lookback,
                                   win_type='gaussian',
                                   min_periods=lookback-1,
                                   center=True
                                   ).mean(std=3)
    # s_lambda = df_lambdas['weighted_average']

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
    return df_rts_conf

def est_rt_wconf(raw_series, lookback, d_infect, remove_outliers=True):
    _gamma = 1 / d_infect

    if remove_outliers:
        raw_series = outlier_removal(raw_series)

    raw_series = raw_series.rolling(lookback, min_periods=lookback-1, center=True).mean()

    pct_series = raw_series.pct_change(fill_method=None)
    pct_series = pct_series.replace([np.inf, -np.inf], np.nan)

    if remove_outliers:
        pct_series = outlier_removal(pct_series)

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
    df_rt = pd.DataFrame(df_rt.stack(), columns=[raw_series.name]).unstack()

    return df_rt

def make_model_dict_state(state_code, abbrev_us_state, df_census, df_st_testing_fmt, covid_params, d_to_forecast = 75,
                        df_mvmt=pd.DataFrame(), df_interventions=pd.DataFrame()):
    model_dict = {}

    model_dict['region_code'] = state_code
    model_dict['region_name'] = abbrev_us_state[model_dict['region_code']]
    model_dict['tot_pop'] = df_census.loc[(df_census.SUMLEV == 40)
                                                  & (df_census.state == model_dict['region_code']),
                                                  'pop2019'].values[0]

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

    model_dict['covid_params'] = covid_params

    # model_dict['df_rts'] = est_all_rts(model_dict)
    # model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['rt_joint_est'].dropna().iloc[0]

    model_dict['df_rts_conf'] = est_all_rts_new(model_dict)
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

    return model_dict

<<<<<<< HEAD
def make_model_dict_us(df_census, df_st_testing_fmt, covid_params, d_to_forecast = 75,
                        df_mvmt=pd.DataFrame(), df_interventions=pd.DataFrame()):
    model_dict = {}

    model_dict['region_code'] = 'US'
    model_dict['region_name'] = 'United States'
    model_dict['tot_pop'] = df_census.loc[(df_census.SUMLEV == 40), 'pop2019'].sum()

    model_dict['df_hist'] = pd.DataFrame()

    model_dict['df_hist']['deaths_tot'] = df_st_testing_fmt['deaths'].sum(axis=1)
    model_dict['df_hist']['deaths_daily'] = model_dict['df_hist']['deaths_tot'].diff()

    model_dict['df_hist']['cases_tot'] = df_st_testing_fmt['cases'].sum(axis=1)
    model_dict['df_hist']['cases_daily'] = model_dict['df_hist']['cases_tot'].diff()
    model_dict['df_hist']['pos_neg_tests_tot'] = df_st_testing_fmt['posNeg'].sum(axis=1)
    model_dict['df_hist']['pos_neg_tests_daily'] = model_dict['df_hist']['pos_neg_tests_tot'].diff()

    model_dict['covid_params'] = covid_params

    # model_dict['df_rts'] = est_all_rts(model_dict)
    # model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['rt_joint_est'].dropna().iloc[0]

    model_dict['df_rts_conf'] = est_all_rts_new(model_dict)
    model_dict['df_rts'] = model_dict['df_rts_conf'].unstack().swaplevel(axis=1)['rt']
    model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['weighted_average'].max()

    model_dict['d_to_forecast'] = int(d_to_forecast)

    if df_mvmt.shape[0] > 0:
        model_dict['df_mvmt'] = df_mvmt
    else:
        model_dict['df_mvmt'] = df_mvmt

    if df_interventions.shape[0] > 0:
        model_dict['df_interventions'] = df_interventions[
            df_interventions.state_code.isin(['US'])].groupby('dt').first().reset_index()
    else:
        model_dict['df_interventions'] = df_interventions

    return model_dict
=======
# def make_model_dict_state(state_code, abbrev_us_state, df_census, df_st_testing_fmt, covid_params, d_to_forecast = 75):
#     model_dict = {}
#
#     model_dict['region_code'] = state_code
#     model_dict['region_name'] = abbrev_us_state[model_dict['region_code']]
#     model_dict['tot_pop'] = df_census.loc[(df_census.SUMLEV == 40)
#                                                   & (df_census.state == model_dict['region_code']),
#                                                   'pop2019'].values[0]
#
#     model_dict['df_hist'] = pd.DataFrame()
#
#     if df_st_testing_fmt['deaths'][model_dict['region_code']].dropna().shape[0] > 14:
#         model_dict['df_hist']['deaths_tot'] = df_st_testing_fmt['deaths'][model_dict['region_code']]
#         model_dict['df_hist']['deaths_daily'] = model_dict['df_hist']['deaths_tot'].diff()
#
#     if df_st_testing_fmt['hospitalizedCurrently'][model_dict['region_code']].dropna().shape[0] > 14:
#         model_dict['df_hist']['hosp_concur'] = df_st_testing_fmt['hospitalizedCurrently'][model_dict['region_code']]
#
#     if df_st_testing_fmt['hospitalizedIncrease'][model_dict['region_code']].dropna().shape[0] > 14:
#         model_dict['df_hist']['hosp_admits'] = df_st_testing_fmt['hospitalizedIncrease'][model_dict['region_code']]
#     model_dict['df_hist']['cases_tot'] = df_st_testing_fmt['cases'][model_dict['region_code']]
#     model_dict['df_hist']['cases_daily'] = model_dict['df_hist']['cases_tot'].diff()
#     model_dict['df_hist']['pos_neg_tests_tot'] = df_st_testing_fmt['posNeg'][model_dict['region_code']]
#     model_dict['df_hist']['pos_neg_tests_daily'] = model_dict['df_hist']['pos_neg_tests_tot'].diff()
#
#     model_dict['covid_params'] = covid_params
#
#     model_dict['df_rts'] = est_all_rts(model_dict)
#
#     model_dict['covid_params']['basic_r0'] = model_dict['df_rts']['rt_joint_est'].dropna().iloc[0]
#
#     model_dict['d_to_forecast'] = int(d_to_forecast)
#
#     return model_dict
>>>>>>> 7be594586e2664f58c707421e53ce1b5ca60d1aa
