import pandas as pd
import numpy as np
from collections import OrderedDict
import io

from coronita_model_helper import outlier_removal

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.dates as mdates
import matplotlib.units as munits

# plt.style.use('fivethirtyeight')
# personalsitestyle = 'fivethirtyeight'
# covidoutlookstyle = 'ggplot'



def add_event_lines(ax, df_interventions):
    curr_xmin, curr_xmax = ax.get_xlim()
    if curr_xmax > 100000:
        return

    df_interventions = df_interventions[
        (df_interventions.dt >= (pd.to_datetime(curr_xmin, unit='d')-pd.Timedelta(days=7)))
        & (df_interventions.dt <= (pd.to_datetime(curr_xmax, unit='d')+pd.Timedelta(days=14)))
    ]

    for thisidx in df_interventions.index:
        if df_interventions.loc[thisidx, 'social_distancing_direction'] == 'holiday':
            thislinecolor = '#8900a5'
        elif df_interventions.loc[thisidx, 'social_distancing_direction'] == 'restricting':
            thislinecolor = '#973200'
        elif df_interventions.loc[thisidx, 'social_distancing_direction'] == 'easing':
            thislinecolor = '#178400'
        ax.axvline(df_interventions.loc[thisidx, 'dt'],
                   color=thislinecolor,
                   linewidth=2.0,
                   linestyle=(0, (1, 4)), #':',
                   alpha=1)
        ax.text(df_interventions.loc[thisidx, 'dt'], ax.get_ylim()[1],
                 df_interventions.loc[thisidx, 'event_name'],
                 rotation=-30, fontsize=10,
                 horizontalalignment='left', verticalalignment='top',
                 color=thislinecolor,
                 alpha=1)

    # ax.set_xlim(min(pd.to_datetime(curr_xmin, unit='d'), df_interventions.dt.min())-pd.Timedelta(days=7),
    #             max(pd.to_datetime(curr_xmax, unit='d'), df_interventions.dt.max())+pd.Timedelta(days=7))
    ax.set_xlim(pd.to_datetime(curr_xmin, unit='d')-pd.Timedelta(days=7),
                pd.to_datetime(curr_xmax, unit='d')+pd.Timedelta(days=14))

    return

def bar_and_line_chart(bar_series, bar_name='', bar_color='#008fd5',
                       line_series = False, line_name='', line_color='#fc4f30',
                       chart_title='', yformat='{:.1%}',
                       bar2_series = None, bar2_name='', bar2_color='#e5ae38', footnote_str=''):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(bar_series.index, bar_series, color=bar_color, label=bar_name)
    if isinstance(bar2_series, pd.Series):
        ax.bar(bar2_series.index, bar2_series, color=bar2_color, label=bar2_name)

    ax.plot(line_series.index, line_series, linestyle='-', color=line_color, label=line_name)

    # set ticks every week
    ax.xaxis.set_major_locator(mpl.dates.WeekdayLocator())
    # set major ticks format
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %d'))

    ax.set_yticklabels([yformat.format(y) for y in ax.get_yticks()])
    plt.xticks(rotation=45)
    ax.set_xlabel('')
    ax.set_title(chart_title)

    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    ax.legend(title='Legend', handles=handles[::-1], labels=labels[::-1])

    
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    return ax

def param_str_maker(model_dict):
    param_dict = {}

    for x, y in model_dict['covid_params'].items():
        if x[-3:] == '_rt':
            param_dict[x] = (y, '{:.2%}')
        else:
            param_dict[x] = (y, '{:.1f}')

    param_fmtd_dict = {}

    for param_name, param_tup in param_dict.items():
        param_val, param_fmt = param_tup
        if type(param_val) == tuple:
            param_fmtd_dict[param_name] = (param_fmt + ' - ' + param_fmt).format(*param_val)
        else:
            param_fmtd_dict[param_name] = param_fmt.format(param_val)

    param_str = '\n'.join(('Parameters Used',
                           r'$D_{{incubation}}: {}$'.format(param_fmtd_dict['d_incub'], ),
                           r'$D_{{infectious}}: {}$'.format(param_fmtd_dict['d_infect'], ),
                           r'$D_{{to hospital}}: {}$'.format(param_fmtd_dict['d_to_hosp'], ),
                           r'$D_{{in hospital}}: {}$'.format(param_fmtd_dict['d_in_hosp'], ),
                           r'$D_{{til death}}: {}$'.format(param_fmtd_dict['d_til_death'], ),
                           r'$Rate_{{Hospitalization}}: {}$'.format(param_fmtd_dict['hosp_rt'], ) + '%',
                           r'$Rate_{{ICU}}: {}$'.format(param_fmtd_dict['icu_rt'], ) + '%',
                           r'$Rate_{{Ventilator}}: {}$'.format(param_fmtd_dict['vent_rt'], ) + '%',
                           r'$Rate_{{Mortality}}: {}$'.format(param_fmtd_dict['mort_rt'], ) + '%',
                           r'$Basic R_{{0}}: {}$'.format(param_fmtd_dict['basic_r0'], )
                           ))
    return param_str

def today_vline(ax):
    ax.axvline(x=pd.Timestamp.today(), ymin=0, ymax=ax.get_ylim()[1], color = 'black', linewidth=1, linestyle=":")
    ax.text(pd.Timestamp.today(), ax.get_ylim()[1]*.95, 'Today: ' + pd.Timestamp.today().strftime("%B %d, %Y"),
             rotation=90, verticalalignment='top', horizontalalignment='right', size='large')
    return ax


### SINGLE REGION CHARTS ###
def ch_exposed_infectious(model_dict):

    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg[['exposed', 'infectious']].dropna(how='all')
    df_chart = df_chart.clip(lower=0)

    ax = df_chart.plot.area(figsize=[14, 8], title=model_dict['region_name']+': Simultaneous Infections Forecast\n'+model_dict['chart_title'],
                            legend=True, color=['#e5ae38', '#fc4f30'])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend(title='Lext Axis Legend', labels=['Exposed Population', 'Infectious Population'],
               loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.set_ylim([0, ax.axes.get_yticks().max()])
    ax.set_yticks(np.linspace(0, ax.axes.get_yticks().max(), 5))
    ax.set_ylabel('People')

    if 'rt_scenario' in model_dict['df_rts'].columns:
        ax2 = ax.twinx()
        r_t = model_dict['df_rts']['rt_scenario'].copy()
        r_t = r_t.reindex(df_chart.index)
        r_t[df_chart.index[0]:df_chart.index[-1]].plot(ax=ax2, color='black', linewidth=2, linestyle='--',
                                                       label=r'Reproduction Factor ($R_t$) - Right Axis', legend=True)
        plt.legend(title='Right Axis Legend', loc="lower right")
        ax2.set_ylim([0, ax2.axes.get_yticks().max()])
        ax2.set_yticks(np.linspace(0, ax2.axes.get_yticks().max(), 5))
        ax2.set_ylabel('$R_t$ Used in Scenario')

        ref_line = r_t[df_chart.index[0]:df_chart.index[-1]].copy()
        ref_line.loc[:] = 1.0
        ref_line.plot(ax=ax2, color=['#008fd5'], legend=True, linewidth=1, label='Reference Line: $R_t = 1$')

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})

    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_cumul_infections(model_dict):

    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg[['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths']].sum(axis=1).dropna(how='all')
    df_chart = df_chart.clip(lower=0)
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Cumulative Infections Forecast\n'+model_dict['chart_title'],
                       legend=True, label='Forecast Cumulative Infections',
                       color=['black'])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if 'cases_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['cases_tot'].loc[df_chart.index[0]:].plot(
            ax=ax, linestyle=':', legend=True, color=['black'],
            label='Reported Cumulative Infections')
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_daily_exposures(model_dict):

    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg['exposed_daily'].dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Daily Exposures Forecast\n'+model_dict['chart_title'],
                       legend=True, color=['#e5ae38'],
                       label='Forecast Daily New Infections (Exposed)')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if 'cases_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['cases_tot'].loc[df_chart.index[0]:].diff().plot(
            ax=ax, linestyle=':', legend=True, color=['#e5ae38'],
            label='Reported Daily New Infections (Exposed)')

    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_hosp(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg[['hospitalized', 'icu', 'vent', 'deaths']].dropna(how='all').copy()
    df_chart = df_chart.rename(columns={'hospitalized':'Forecast Concurrent Hospitalizations',
                                        'icu':'Forecast ICU Cases',
                                        'vent':'Forecast Ventilations',
                                        'deaths':'Forecast Cumulative Deaths'})

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Hospitalization and Deaths Forecast\n'+model_dict['chart_title'],
                       color=['#6d904f', '#8b8b8b', '#810f7c', '#fc4f30'],
                       label=['Forecast Concurrent Hospitalizations',
                              'Forecast ICU Cases',
                              'Forecast Ventilations',
                              'Forecast Cumulative Deaths'])
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    if 'hosp_concur' in model_dict['df_hist'].columns:
        model_dict['df_hist']['hosp_concur'].plot(ax=ax, linestyle=':', legend=True,
                                                  label='Reported Concurrent Hospitalizations',
                                                  color=['#6d904f'])
    if 'deaths_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['deaths_tot'].loc[
        df_chart.index[0]:].plot(ax=ax, linestyle=':', legend=True,
                                 label='Reported Total Deaths',
                                 color='#fc4f30')
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_hosp_admits(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg['hosp_admits'].dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Daily Hospital Admissions Forecast\n'+model_dict['chart_title'],
                       label='Forecast Hospital Admissions',
                       color='#6d904f')
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    if 'hosp_admits' in model_dict['df_hist'].columns:
        model_dict['df_hist']['hosp_admits'].plot(ax=ax, linestyle=':', legend=True,
                                                  label='Reported Hospital Admissions',
                                                  color='#6d904f'
                                                  )
        if model_dict['df_hist']['hosp_admits'].max() > outlier_removal(model_dict['df_hist']['hosp_admits']).max():
            ax.set_ylim([0, 1.1 * max(outlier_removal(model_dict['df_hist']['hosp_admits']).max(), df_chart.max())])

    ax.set_ylim([0, max(ax.get_ylim()[1], 5)])

    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_daily_deaths(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg['deaths'].diff().dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Daily Deaths Forecast\n'+model_dict['chart_title'],
                       label='Forecast Daily Deaths',
                       color='#fc4f30')

    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    if 'deaths_daily' in model_dict['df_hist'].columns:
        model_dict['df_hist']['deaths_daily'].plot(ax=ax, linestyle=':', legend=True,
                                                  label='Reported Daily Deaths',
                                                  color='#fc4f30'
                                                  )
        if model_dict['df_hist']['deaths_daily'].max() > outlier_removal(model_dict['df_hist']['deaths_daily']).max():
            ax.set_ylim([0, 1.1 * max(outlier_removal(model_dict['df_hist']['deaths_daily']).max(), df_chart.max())])

    ax.set_ylim([0, max(ax.get_ylim()[1], 5)])

    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_doubling_rt(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    ## DOUBLING RATE CHART
    df_chart = np.log(2) / df_agg[['exposed_daily', 'hospitalized', 'deaths']].pct_change().rolling(7, center=True).mean().dropna(how='all')
    df_chart = df_chart.mask(df_chart < 0)
    df_chart = df_chart.rename(columns={'exposed_daily':'New Cases',
                                        'hospitalized':'Hospitalizations',
                                        'deaths':'Daily Deaths'})
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot(figsize=[14, 8], title=model_dict['region_name']+': Doubling Rate Forecast\n' + model_dict['chart_title'],
                       color=['#e5ae38', '#6d904f', '#fc4f30'])

    if 'cases_daily' in model_dict['df_hist'].columns:
        case_dr = np.log(2) / model_dict['df_hist']['cases_daily'].pct_change().rolling(7, center=True).mean()
        case_dr.mask(case_dr < 0).plot(ax=ax, linestyle=':', legend=True, color=['#e5ae38'],
                     label='Reported New Cases')

    if 'hosp_concur' in model_dict['df_hist'].columns:
        hosp_dr = np.log(2) / model_dict['df_hist']['hosp_concur'].pct_change().rolling(7, center=True).mean()
        hosp_dr.mask(hosp_dr < 0).plot(ax=ax, linestyle=':', legend=True, color=['#6d904f'],
                     label='Reported Concurrent Hospitalizations')

    if 'deaths_tot' in model_dict['df_hist'].columns:
        deaths_dr = np.log(2) / model_dict['df_hist']['deaths_tot'].pct_change().rolling(7, center=True).mean()
        deaths_dr.mask(deaths_dr < 0).plot(ax=ax, linestyle=':', legend=True, color=['#fc4f30'],
                       label='Reported Total Deaths')
    plt.yscale('log')
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    ax.set_ylabel('Days til Doubling')
    ax.set_ylim(1, 1000)
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_population_share(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = df_agg[['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered']].dropna(how='all')
    df_chart = df_chart.rename(columns={'susceptible':'Forecast Susceptible Population',
                                        'exposed':'Forecast Exposures',
                                        'infectious':'Forecast Infectious',
                                        'hospitalized':'Forecast Hospitalizations',
                                        'recovered':'Forecast Recoveries',
                                        'deaths':'Forecast Deaths'})
    df_chart = df_chart.clip(lower=0)
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot.area(figsize=[14, 8], title=model_dict['region_name']+': Population Overview Forecast\n'+model_dict['chart_title'])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax2 = ax.twinx()
    pd.Series(1.0, index=df_chart.index).plot(ax=ax2, color='black', linewidth=0, linestyle='--', legend=False)

    ax.set_ylim([0, df_chart.sum(axis=1).max()])
    ax.set_yticks(np.linspace(0, df_chart.sum(axis=1).max(), 5))
    ax.set_ylabel('Population')
    ax2.set_ylim([0, 1.0])
    ax2.set_yticks(np.linspace(0, 1.0, 5), minor=False)
    ax2.set_yticks(np.linspace(0, 1.0, 20), minor=True) # Replaced 25 with 20
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax2.tick_params(axis='y', reset=True, direction='inout', which='minor', color='black', left=True,
                    length=10 , width=1)
    plt.box(on=None)
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    ax = today_vline(ax)
    return ax

def ch_rts(model_dict):
    param_str = param_str_maker(model_dict)

    # solo_rts = [x for x in model_dict['df_rts'].columns if x in
    #             ['rt_deaths_daily', 'rt_hosp_concur', 'rt_hosp_admits', 'rt_pos_test_share_daily', 'rt_cases_daily']]

    df_just_rts = model_dict['df_rts'].dropna(how='all')

    solo_rts = [x for x in df_just_rts.columns if x in
                ['test_share', 'deaths_daily', 'hosp_concur', 'hosp_admits',
                 'cases_daily'] ]
    ax = df_just_rts[solo_rts].dropna(how='all').plot(figsize=[14, 8], alpha=0.2,
                                                               title=model_dict['region_name']+r': Reproduction Rate ($R_{t}$) Estimates',
                                                               legend=True)

    df_just_rts['weighted_average'].dropna().plot(ax=ax, legend=True)

    ax.set_ylim([0,ax.get_ylim()[1]])

    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1, title=model_dict['region_name']+': Reproduction Factor Estimates')

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_rt_confid(model_dict):
    df_rt = model_dict['df_rts_conf'][['weighted_average']].unstack('metric')
    param_str = param_str_maker(model_dict)
    rt_name = df_rt.columns.levels[0][0]
    df_rt = df_rt[rt_name].dropna(how='all')

    # lower68, upper68, lower95, upper95 = df_rt.iloc[:,1:5]
    ax = df_rt['rt'].dropna(how='all').plot(figsize=[14, 8],
                                     title=model_dict['region_name']+r': Reproduction Rate ($R_{t}$) Estimate'+'\n'+model_dict['chart_title'],
                                     legend=True,
                                            label='$R_t$: {}'.format(rt_name))
    ci68 = plt.fill_between(df_rt['rt_u68'].index, df_rt['rt_u68'], df_rt['rt_l68'],
                            alpha=0.5, facecolor='#e5ae38', label='68% Confidence Interval')
    ci95 = plt.fill_between(df_rt['rt_u95'].index, df_rt['rt_u95'], df_rt['rt_l95'],
                            alpha=0.25, facecolor='#e5ae38', label='95% Confidence Interval')
    ax.set_ylim([0,df_rt['rt'].max()+0.5])
    ax.set_xlim([df_rt['rt'].dropna().index.min(), df_rt['rt'].dropna().index.max()])

    ref_line = df_rt['rt'].dropna().copy()
    ref_line.loc[:] = 1.0
    ref_line.plot(ax=ax, color=['black'], legend=True, linewidth=1, label='Reference Line: $R_t = 1$')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    return ax

def ch_totaltests(model_dict):
    ax = bar_and_line_chart(bar_series=model_dict['df_hist']['pos_neg_tests_daily'].dropna(how='all'),
                       bar_name='# of Negative Tests',
                       line_series=model_dict['df_hist']['pos_neg_tests_daily'].rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average', yformat='{:0,.0f}',
                       chart_title='{}: Total COVID-19 Tests Per Day'.format(model_dict['region_name']),
                       bar2_series=model_dict['df_hist']['cases_daily'], bar2_name='# of Positive Tests',
                            footnote_str=model_dict['footnote_str']
                       )
    return ax

def ch_positivetests(model_dict):
    ax = bar_and_line_chart(bar_series=model_dict['df_hist']['cases_daily'].dropna(how='all'),
                       bar_name='# of Positive Tests', bar_color='#e5ae38',
                       line_series=model_dict['df_hist']['cases_daily'].rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average', yformat='{:0,.0f}',
                       chart_title='{}: Positive COVID-19 Tests Per Day'.format(model_dict['region_name']),
                            footnote_str=model_dict['footnote_str']
                       )
    return ax

def ch_postestshare(model_dict):
    df_chart = model_dict['df_hist'][['cases_daily', 'pos_neg_tests_daily']].clip(lower=0)
    df_chart['neg_tests_daily'] = (df_chart['pos_neg_tests_daily'] - df_chart['cases_daily']).clip(lower=0)
    df_chart = df_chart.div(df_chart[['cases_daily', 'pos_neg_tests_daily']].max(axis=1), axis=0).dropna(how='all')
    df_chart['sevendayavg'] = df_chart['cases_daily'].mask(df_chart['cases_daily'] >= 1.0
                                                           ).rolling(7, min_periods=1).mean()

    ax = df_chart[['cases_daily', 'neg_tests_daily']].plot(
        kind='area', stacked=True,
        title='{}: COVID-19 Positivity Rate'.format(model_dict['region_name']),
        figsize=[14, 8], color=['#e5ae38', '#008fd5'])
    df_chart['sevendayavg'] = df_chart['cases_daily'].mask(df_chart['cases_daily']>=1.0
                                                           ).rolling(7, min_periods=1).mean()
    df_chart['sevendayavg'].plot(ax=ax, linestyle='-', color='#fc4f30')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.legend(title='Legend', bbox_to_anchor=(1.07, 0.95), ncol=1,
               labels=['% Positive Tests', '% Negative Tests', '7-Day Rolling Average - Positivity Rate'])
    ax.set_ylim(0, min(1, df_chart['sevendayavg'].max() * 1.25))
    ax.set_xlim(ax.get_xlim()[0] + 10, ax.get_xlim()[1] - 10)
    ax.set_xlabel('')

    inset_days = 90

    if df_chart.iloc[inset_days*-1:]['sevendayavg'].max() < (0.15 * ax.get_ylim()[1]):
        axins = inset_axes(ax, width="40%", height="60%", loc=4, borderpad=4,
                          axes_kwargs={'alpha': 1, 'frame_on': True})

        df_chart[['cases_daily', 'neg_tests_daily']].plot(
            ax=axins, kind='area', stacked=True,
            color=['#e5ae38', '#008fd5'], legend=False)
        df_chart['sevendayavg'].plot(ax=axins, linestyle='-', color='#fc4f30', legend=False)

        if df_chart['sevendayavg'].iloc[-inset_days:].max() * 1.1 < .1:
            axins.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1%}'))
        else:
            axins.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))

        axins.set_ylim(0, min(1, df_chart.iloc[inset_days*-1:]['sevendayavg'].max() * 1.25))
        axins.set_xlim(axins.get_xlim()[1]-inset_days-10, axins.get_xlim()[1] - 10)
        axins.set_xlabel('')

        months = mdates.MonthLocator()  # every month
        months_fmt = mdates.DateFormatter('%b')
        axins.xaxis.set_major_locator(months)
        axins.xaxis.set_major_formatter(months_fmt)
        mark_inset(ax, axins, loc1=3, loc2=4)

    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_googmvmt(model_dict):
    df_chart = model_dict['df_mvmt']
    df_chart = df_chart[['retail_and_recreation_percent_change_from_baseline',
         'grocery_and_pharmacy_percent_change_from_baseline',
         'parks_percent_change_from_baseline',
         'transit_stations_percent_change_from_baseline',
         'workplaces_percent_change_from_baseline',
         'residential_percent_change_from_baseline']]
    df_chart = df_chart.interpolate(limit_area='inside').rolling(7).mean().div(100.0)

    labels = [x[:-29].title().replace('_', ' ') for x in df_chart.columns]

    ax = df_chart.plot(title='{}: Google Movement Data\nRolling 7-day Average'.format(model_dict['region_name']),
                       figsize=[14, 8], )
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    plt.legend(title='Percent Change from Baseline', bbox_to_anchor=(1.07, 0.95), ncol=1, labels=labels)
    ax.set_xlabel('')

    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_detection_rt(model_dict):
    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']

    df_chart = model_dict['df_hist']['cases_daily'].rolling(7).sum().div(
        df_agg['exposed_daily'].rolling(7).sum()).dropna()

    df_chart = df_chart.rolling(7, win_type='gaussian', center=True, min_periods=1).mean(std=3)

    ax = df_chart.plot(
        figsize=[14, 8],
        title=model_dict['region_name']+': COVID-19 Daily Infection Detection Rate\n' + model_dict['chart_title'],
        legend=False
    )
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_ylim([0, ax.get_ylim()[1]])

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')

    plt.annotate(model_dict['footnote_str'],
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    return ax

### MULTI-REGION CHARTS ###
def ch_rt_summary(df_wavg_rt_conf_allregs):
    import plotly.express as px

    df_chart = df_wavg_rt_conf_allregs.unstack('metric').apply(
        lambda x: x.loc[x.last_valid_index()]).sort_values(ascending=False).unstack('metric')
    df_chart['e'] = df_chart.rt_u68 - df_chart.rt
    df_chart = df_chart.round(2)
    df_chart = df_chart.sort_values(by='rt').reset_index().rename(columns={'index': 'state'})

    fig = px.scatter(df_chart,
                     y="state", x="rt",
                     error_x="e", color='rt',
                     color_continuous_scale=px.colors.diverging.PiYG_r,
                     color_continuous_midpoint=1.0,
                     height=1000
                     )

    fig.update_traces(mode='markers+text',
                      marker_line_width=1,
                      marker_size=13,
                      text=pd.Series(df_chart['state']).apply(
                          lambda
                              x: "<a href='http://www.michaeldonnel.ly/covid19/datacenter/{0}/' style='color: black'>{0}</a>".format(
                              x)).to_list(),
                      textfont=dict(size=8)
                      )
    fig.update_layout(title='COVID-19: Current Estimated Reproduction Factor',
                      yaxis=dict(fixedrange=True),
                      xaxis=dict(fixedrange=True)
                      )
    fig.update_xaxes(title_text='Effective Reproduction Factor')
    fig.update_yaxes(title_text='State')
    return fig

def ch_exposure_prob(df_fore_allstates, s_pop):
    infectious_contact_prob = df_fore_allstates.loc[
        pd.Timestamp.today().date()].loc[['exposed', 'infectious']].sum().div(
        s_pop).sort_values()

    colorbar_name = 'Probability'

    df_chart = (1 - (1 - infectious_contact_prob) ** 10).reset_index()
    df_chart.columns = ['state', colorbar_name]
    df_chart[colorbar_name] = df_chart[colorbar_name].mul(100).round(1)

    import plotly.express as px

    chart_title = 'US: Current Model-Estimated COVID-19 Exposure Probability Per 10 Contacts'

    fig = px.choropleth(df_chart[['state', colorbar_name]],
                        locations=df_chart['state'],
                        locationmode="USA-states",
                        color=colorbar_name,
                        color_continuous_scale="BuPu",
                        title=chart_title,
                        projection='albers usa',
                        )
    fig.update_layout(autosize=True,
            margin=dict(l=10,r=10,b=100,t=100, pad=0),
        coloraxis_colorbar=dict(len=0.75,thickness=30,
                                yanchor="top", y=1,
                                ticks="outside", ticksuffix="%")
        )

    return fig

def ch_exposure_prob_anim(df_fore_allstates, df_census):
    from covid_data_helper import abbrev_us_state
    s_pop = df_census[df_census.SUMLEV == 40].set_index('state')['pop2019']
    colorbar_name = 'Probability'
    df_chart = df_fore_allstates.unstack('dt').loc[['exposed', 'infectious']].T.sum(axis=1)
    df_chart = df_chart.unstack(0).div(s_pop).dropna(how='all', axis=1)
    df_chart = df_chart.loc[:pd.Timestamp.today()].stack()
    df_chart = (1 - (1 - df_chart) ** 10).reset_index()
    df_chart.columns = ['dt', 'state', colorbar_name]
    df_chart['fullstatename'] = df_chart['state'].replace(abbrev_us_state)
    df_chart[colorbar_name] = df_chart[colorbar_name].mul(100).round(1)
    # df_firstframe = df_chart[df_chart.dt == df_chart.dt.max()].copy()
    # df_firstframe['dt'] = 'Today (' + df_firstframe.dt.dt.strftime('%B %d, %Y') + ')'
    df_chart = df_chart[df_chart.dt.isin(
        pd.date_range(
            end=pd.Timestamp.today().normalize(),
            start=pd.Timestamp('2020-03-01'),
            # periods=100,
            freq='3d'
        ).append( pd.Index([pd.Timestamp.today().normalize()]) )
    ) ]
    df_chart['dt'] =df_chart.dt.dt.strftime('%B %d, %Y')
    # df_chart = pd.concat([df_firstframe,df_chart])

    import plotly.express as px
    import plotly.graph_objects as go

    chart_title = 'US: Model-Estimated COVID-19 Exposure Probability Per 10 Contacts'

    fig = px.choropleth(df_chart,
                        locations=df_chart['state'],
                        locationmode="USA-states",
                        color=colorbar_name,
                        color_continuous_scale="BuPu",
                        range_color=(0, 10),
                        title=chart_title,
                        projection='albers usa',
                        animation_group='state',
                        animation_frame='dt',
                        hover_name='fullstatename',
                        hover_data=[colorbar_name]
                        )

    fig.update_layout(autosize=True)

    fig = go.Figure(fig)

    fig_dict = fig.to_dict()

    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 0, "redraw": True},
                                    "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 50, 'b': 10},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    fig_dict["layout"]["sliders"][0]['currentvalue'] = {
        "font": {"size": 20, "family": 'Roboto'},
        "prefix": "<b>Date: ", "suffix": "</b>",
        "visible": True,
        "xanchor": "right"
    }
    fig_dict["layout"]["sliders"][0]['len'] = 0.95
    fig_dict['layout']['sliders'][0]['pad'] = {'b': 10, 't': 20, 'l': 10, 'r': 10}
    fig_dict['layout']['sliders'][0]['transition'] = {"duration": 0, "easing": "linear"}
    fig_dict['layout']['margin'] = {'b': 130, 'l': 0, 'r': 0, 't': 30}
    fig_dict['layout']['coloraxis']['colorbar'] = {
        'thickness': .035, 'thicknessmode': 'fraction', 'xpad': 5, 'ypad': 5,
        'len': 0.75, 'lenmode': 'fraction',
        # 'title':'Exposure Probability',
        'tickvals':[0, 2, 4, 6, 8, 10],
        'ticktext':['0%', '2%', '4%', '6%', '8%', '10%+']}

    for frame in range(len(fig.frames)):
        this_ht = fig_dict['frames'][frame]['data'][0]['hovertemplate'].replace('<br>dt=', '')
        this_ht = this_ht.replace('<br>state=%{location}<extra></extra>', '%')
        this_ht = this_ht.replace('Probability=', 'Exposure Probability: ')
        fig_dict['frames'][frame]['data'][0]['hovertemplate'] = this_ht


    fig = go.Figure(fig_dict)
    fig.layout.sliders[0]['active'] = len(fig.frames) - 1
    fig.update_traces(z=fig.frames[-1].data[0].z,
                     hovertemplate=fig.frames[-1].data[0].hovertemplate)

    return fig

def ch_statemap(df_chart, region_name, scope=['USA']):
    import plotly.express as px
    import plotly.figure_factory as ff

    chart_title = region_name + ': COVID-19 Cases Per 100k Residents'

    fig = ff.create_choropleth(
        fips=df_chart['fips'],
        values=df_chart['cases_per100k'], show_state_data=True,
        scope=scope,  # Define your scope
        round_legend_values=True,
        colorscale=px.colors.sequential.amp,
        binning_endpoints=list(np.linspace(0, 1000, 11)),
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
        state_outline={'color': 'black', 'width': 1.0},
        legend_title='cases_per100k', title=chart_title,
        width=800, height=400,
        font=dict(size=12)
    )

    return fig

def ch_statemap2(df_chart, region_name, scale_max, counties, fitbounds='locations'):
    import plotly.express as px

    chart_title = region_name + ': COVID-19 Cases Per 100k Residents'

    fig = px.choropleth(df_chart.reset_index()[['fips','county','cases_per100k']].dropna(),
                        geojson=counties,
                        locations='fips', color='cases_per100k',
                        color_continuous_scale="amp",
                        range_color=(0, max((scale_max//200),1)*200),
                        scope="usa",
                        title=chart_title,
                        projection='albers usa',
                        hover_name='county',
                        hover_data=['cases_per100k'],
                        labels={'cases_per100k':'Cases Per 100k'}
                        )
    fig.update_traces(marker_line_width=0.5, marker_opacity=1.0, marker_line_color='gray')

    if region_name == 'Alaska':
        fitbounds = False
    fig.update_geos(
        fitbounds=fitbounds ,
                    visible=True,
                    showsubunits=True,
                    subunitcolor="black"
    )

    return fig

def ch_statemap_casechange(model_dict, df_counties, counties_geo, fitbounds='locations'):
    region_name = model_dict['region_name']

    df_chart = df_counties['cases_per100k']
    df_chart = df_chart.unstack(['state', 'county', 'fips']).diff(periods=14)
    df_chart = df_chart.dropna(how='all', axis=1)
    df_chart = df_chart.apply(lambda x: x[x.last_valid_index()]).reset_index()
    df_chart = df_chart.rename(columns={0: 'cases_norm_14d_chg'})
    df_chart['county'] = df_chart['county'] + ', ' + df_chart['state']

    scale_max = df_chart.cases_norm_14d_chg.quantile(.9)

    if model_dict['region_code'] != 'US':
        df_chart = df_chart[df_chart.state == model_dict['region_code']]

    import plotly.express as px

    chart_title = region_name + ': New COVID-19 Cases Per 100k Residents Over Last 14 Days'

    fig = px.choropleth(df_chart,
                        geojson=counties_geo,
                        locations='fips', color='cases_norm_14d_chg',
                        color_continuous_scale="amp",
                        range_color=(0, max((scale_max//200),1)*200),
                        scope="usa",
                        title=chart_title,
                        projection='albers usa',
                        hover_name='county',
                        hover_data={'cases_norm_14d_chg': ':.1f'},
                        labels={'cases_norm_14d_chg':'New Cases Per 100k'}
                        )


    if model_dict['region_code'] in ['US','AK']:
        fitbounds = False
        marker_line_width = 0.1
    else:
        marker_line_width = 0.25

    fig.update_traces(marker_line_width=marker_line_width, marker_opacity=1.0, marker_line_color='gray')
    fig.update_geos(fitbounds=fitbounds, visible=True, showsubunits=True, subunitcolor="black")

    fig_dict = fig.to_dict()
    fig_dict['layout']['margin'] = {'b': 50, 'l': 0, 'r': 0, 't': 30}
    fig_dict['layout']['coloraxis']['colorbar'] = {'thickness': .035, 'thicknessmode': 'fraction', 'xpad': 5, 'ypad': 5,
                                                   'len': 0.75, 'lenmode': 'fraction'}
    fig_dict['data'][0]['hovertemplate'] = fig_dict['data'][0]['hovertemplate'].replace(
        '<br><br>New Cases Per 100k=', '<br>New Cases Per 100k: ').replace(
        '<br>fips=%{location}<extra></extra>', ''
    )
    import plotly.graph_objects as go
    fig = go.Figure(fig_dict)
    fig.update_layout(autosize=True)

    if model_dict['region_code'] in ['US']:
        fig.update_traces(marker_line_width=0, marker_opacity=0.8)
        fig.update_geos(showsubunits=True, subunitcolor="black")

    return fig

def ch_statemap_casechange_anim(model_dict, df_counties, counties_geo, fitbounds='locations'):
    region_name = model_dict['region_name']
    df_chart = df_counties['cases_per100k']
    df_chart = df_chart.unstack(['state', 'county', 'fips']).fillna(0).diff(periods=14)
    df_chart = df_chart.dropna(how='all', axis=1)
    df_chart = df_chart.unstack('dt').reset_index().dropna()
    df_chart = df_chart.rename(columns={0: 'cases_norm_14d_chg'})
    # df_chart[(df_chart.dt + pd.Timedelta(days=1)).dt.day == 1] ## Last day of the month
    df_chart = df_chart[df_chart.dt.dt.day == df_chart.dt.max().day]
    df_chart['dt'] =df_chart.dt.dt.strftime('%B %d, %Y')

    scale_max = df_chart.cases_norm_14d_chg.quantile(.9)

    if model_dict['region_code'] != 'US':
        df_chart = df_chart[df_chart.state == model_dict['region_code']]

    import plotly.express as px

    chart_title = region_name + ': 14-Day Change in COVID-19 Cases Per 100k Residents'

    fig = px.choropleth(df_chart,
                        geojson=counties_geo,
                        locations='fips', color='cases_norm_14d_chg',
                        color_continuous_scale="amp",
                        range_color=(0, max((scale_max//200),1)*200),
                        scope="usa",
                        title=chart_title,
                        projection='albers usa',
                        animation_group='fips',
                        animation_frame='dt',
                        hover_name='county',
                        hover_data=['cases_norm_14d_chg'],
                        labels={'cases_norm_14d_chg':'14-day Change'}
                        )
    if model_dict['region_code'] in ['US','AK']:
        fitbounds = False
        marker_line_width = 0.1
    else:
        marker_line_width = 0.25

    fig.update_traces(marker_line_width=marker_line_width, marker_opacity=1.0, marker_line_color='gray',
                      hovertemplate=None)
    fig.update_geos(fitbounds='locations', visible=True, showsubunits=True, subunitcolor="black")
    return fig

def calc_trend(series, threshold):
    df = series.rolling(14).mean() - series.rolling(28).mean().fillna(0)
    df = pd.cut(df.stack(), [-np.inf, -1*threshold, threshold, np.inf], labels=['▼','','▲'])
    return df.unstack().iloc[-1]

def tab_summary(df_st_testing_fmt, df_fore_allstates, df_census, df_wavg_rt_conf_allregs):
    df_tab = df_census[df_census.SUMLEV == 40].copy()
    df_tab = df_tab.set_index('state')

    # Deaths
    df_tab['Total Deaths'] = df_st_testing_fmt['deaths'].fillna(method='ffill').iloc[-1]
    df_tab['Total Deaths per 100k'] = df_tab['Total Deaths'].div(df_tab.pop2019).mul(1e5)
    df_tab['14-Day Avg Daily Deaths'] = df_st_testing_fmt['deaths'].fillna(method='ffill').diff().rolling(14).mean().iloc[-1]
    df_tab['14-Day Avg Daily Deaths per 100k'] = df_tab['14-Day Avg Daily Deaths'].div(df_tab.pop2019).mul(1e5)
    df_tab['deaths_trend'] = calc_trend(
        df_st_testing_fmt['deaths'].fillna(method='ffill').div(df_tab.pop2019).mul(1e5).diff(),
        0.02)

    # Cases
    df_tab['Total Cases'] = df_st_testing_fmt['cases'].fillna(method='ffill').iloc[-1]
    df_tab['Total Cases per 100k'] = df_tab['Total Cases'].div(df_tab.pop2019).mul(1e5)
    df_tab['14-Day Avg Daily Cases'] = df_st_testing_fmt['cases'].fillna(method='ffill') \
        .diff().rolling(14).mean().iloc[-1]
    df_tab['14-Day Avg Daily Cases per 100k'] = df_tab['14-Day Avg Daily Cases'].div(df_tab.pop2019).mul(1e5)
    df_tab['cases_trend'] = calc_trend(
        df_st_testing_fmt['cases'].fillna(method='ffill').div(df_tab.pop2019).mul(1e5).diff(),
        0.5)
    # Positivity Rate
    df_tab['Positivity Rate'] = df_st_testing_fmt['cases'].diff().rolling(14).sum().iloc[-1].div(
        df_st_testing_fmt['posNeg'].diff().rolling(14).sum().iloc[-1])
    df_tab['positivity_trend'] = calc_trend(
        df_st_testing_fmt['cases'].diff().rolling(14).sum().div(
            df_st_testing_fmt['posNeg'].diff().rolling(14).sum()),
        0.005)
    # Hospitalizations
    df_tab['Hospitalized'] = df_st_testing_fmt['hospitalizedCurrently'].fillna(method='ffill').iloc[-1]
    df_tab['Hospitalized per 100k'] = df_tab['Hospitalized'].div(df_tab.pop2019).mul(1e5)
    df_tab['hospconcur_trend'] = calc_trend(
        df_st_testing_fmt['hospitalizedCurrently'].fillna(method='ffill').div(df_tab.pop2019).mul(1e5),
        0.5)
    df_tab['14-Day Avg Daily Hosp Admits'] = df_st_testing_fmt['hospitalizedCumulative'].diff().fillna(method='ffill') \
        .rolling(14).mean().iloc[-1]
    df_tab['14-Day Avg Daily Hosp Admits per 100k'] = df_tab['14-Day Avg Daily Hosp Admits'].div(df_tab.pop2019).mul(1e5)
    df_tab['hospadmits_trend'] = calc_trend(
        df_st_testing_fmt['hospitalizedCumulative'].diff().fillna(method='ffill').div(df_tab.pop2019).mul(1e5),
        0.05)

    df_tab.loc[:, [col for col in df_tab.columns if '_trend' in col]] = df_tab.loc[:, [col for col in df_tab.columns if '_trend' in col]].fillna('')

    # Modeled
    df_tab['Model Est\'d Active Infections'] = \
        df_fore_allstates[[col for col in df_fore_allstates.columns if col != 'US']] \
            .unstack('dt').loc[['exposed', 'infectious']].T.sum(axis=1) \
            .unstack(0).loc[pd.Timestamp.today().date()]
    df_tab['Model Est\'d Active Infections per 100k'] = df_tab['Model Est\'d Active Infections'].div(df_tab.pop2019).mul(1e5)
    df_tab = df_tab.sort_values(by='Model Est\'d Active Infections per 100k', ascending=False)
    df_tab['Current Reproduction Rate (Rt)'] = df_wavg_rt_conf_allregs.unstack('metric').swaplevel(axis=1)['rt'].fillna(method='ffill').iloc[-1]

    df_tab = df_tab.reset_index()
    df_tab['Riskiest State Rank'] = df_tab.index + 1

    # Formatting
    # df_tab['State'] = df_tab.county  # + ' (' + df_tab.state + ')'
    df_tab['State'] = '<a href="/forecasts/' + df_tab.state + '" target="_top">' + df_tab.county + '</a>'
    dict_col_names = {'pop2019': 'Population'}
    df_tab = df_tab.rename(columns=dict_col_names)

    ## US Table ##
    df_tab_us = pd.DataFrame(df_tab.sum(skipna=False)).T
    df_tab_us['Positivity Rate'] = (df_st_testing_fmt['cases'].diff().rolling(14).sum().sum(axis=1).iloc[-1]
                                    / df_st_testing_fmt['posNeg'].diff().rolling(14).sum().sum(axis=1).iloc[-1])
    df_tab_us['Current Reproduction Rate (Rt)'] = df_wavg_rt_conf_allregs.unstack('metric').swaplevel(axis=1)['rt'].fillna(method='ffill')['US'].iloc[-1]
    ##############

    format_dict = {
        'Riskiest State Rank': '{0:.0f}',
        'Population': '{0:,.0f}',
        'Model Est\'d Active Infections per 100k': '{0:,.0f}',
        'Current Reproduction Rate (Rt)': '{0:.2f}',
        'Total Cases per 100k': '{0:,.0f}',
        '14-Day Avg Daily Cases per 100k': '{0:,.1f}',
        'Positivity Rate': '{:.1%}',
        'Total Deaths per 100k': '{0:,.0f}',
        '14-Day Avg Daily Deaths per 100k': '{0:,.1f}',
        'Hospitalized per 100k': '{0:,.2f}',
        '14-Day Avg Daily Hosp Admits per 100k': '{0:,.2f}',
        'Model Est\'d Active Infections': '{0:,.0f}',
        'Total Cases': '{0:,.0f}',
        '14-Day Avg Daily Cases': '{0:,.1f}',
        'Total Deaths': '{0:,.0f}',
        '14-Day Avg Daily Deaths': '{0:,.1f}',
        'Hospitalized': '{0:,.0f}',
        '14-Day Avg Daily Hosp Admits': '{0:,.2f}'
                   }

    for key, value in format_dict.items():
        df_tab[key] = df_tab[key].map(value.format)
        df_tab_us[key] = df_tab_us[key].map(value.format)

    ## Add Trend Arrows ##
    df_tab['14-Day Avg Daily Deaths per 100k'] = df_tab['14-Day Avg Daily Deaths per 100k'] \
                                                 + df_tab['deaths_trend'].astype(str)
    df_tab['14-Day Avg Daily Cases per 100k'] = df_tab['14-Day Avg Daily Cases per 100k'] \
                                                + df_tab['cases_trend'].astype(str)
    df_tab['Positivity Rate'] = df_tab['Positivity Rate'] \
                                + df_tab['positivity_trend'].astype(str)
    df_tab['Hospitalized per 100k'] = df_tab['Hospitalized per 100k'] \
                                      + df_tab['hospconcur_trend'].astype(str)
    df_tab['14-Day Avg Daily Hosp Admits per 100k'] = df_tab['14-Day Avg Daily Hosp Admits per 100k'] \
                                                      + df_tab['hospadmits_trend'].astype(str)
    df_tab['14-Day Avg Daily Deaths'] = df_tab['14-Day Avg Daily Deaths'] \
                                                 + df_tab['deaths_trend'].astype(str)
    df_tab['14-Day Avg Daily Cases'] = df_tab['14-Day Avg Daily Cases'] \
                                                + df_tab['cases_trend'].astype(str)
    df_tab['Hospitalized'] = df_tab['Hospitalized'] \
                                      + df_tab['hospconcur_trend'].astype(str)
    df_tab['14-Day Avg Daily Hosp Admits'] = df_tab['14-Day Avg Daily Hosp Admits'] \
                                                      + df_tab['hospadmits_trend'].astype(str)
    ######################

    output_cols = ['Riskiest State Rank', 'State', 'Population',
                   'Model Est\'d Active Infections per 100k', 'Current Reproduction Rate (Rt)',
                   'Total Cases per 100k', '14-Day Avg Daily Cases per 100k',
                   'Positivity Rate',
                   'Total Deaths per 100k', '14-Day Avg Daily Deaths per 100k',
                   'Hospitalized per 100k', '14-Day Avg Daily Hosp Admits per 100k'
                   ]

    tab_html = df_tab[output_cols].replace('nan','Not Available').to_html(index=False, border=0, justify='center', escape=False)
    tab_html = tab_html.replace('<table', '<table class="w3-table w3-striped w3-hoverable w3-medium sortable"')

    tab_header = "---\nlayout: noheader\n---\n"
    tab_html = tab_header + '<script src="/assets/js/sorttable.js" type="text/javascript"></script>' \
               + tab_html
    tab_html = tab_html.replace('▼', '<span style="color: green">▼</span>').replace('▲',
                                                                                    '<span style="color: red">▲</span>')

    return tab_html, df_tab, df_tab_us


def run_all_charts(model_dict, scenario_name='', pdf_out=False, show_charts=True, pub2web=False):
    model_dict['chart_title'] = "{0}: {1} Scenario".format(
        model_dict['region_name'], scenario_name)

    param_str = param_str_maker(model_dict)
    df_agg = model_dict['df_agg']
    event_lines = model_dict['df_interventions']

    if pdf_out and type(pdf_out) == str:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = PdfPages(pdf_out)
    elif pdf_out:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = pdf_out

    thischart = 'ch_rt_confid'
    ax = ch_rt_confid(model_dict)
    if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_rts'
    ax = ch_rts(model_dict)
    if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    if model_dict['df_mvmt'].shape[0] > 0:
        thischart = 'ch_googmvmt'
        ax = ch_googmvmt(model_dict)
        if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
        if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
        if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
        if show_charts:
            plt.show()
        else:
            plt.close()

    thischart = 'ch_exposed_infectious'
    ax = ch_exposed_infectious(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_hosp'
    ax = ch_hosp(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()
    plt.close()

    thischart = 'ch_population_share'
    ax = ch_population_share(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_cumul_infections'
    ax = ch_cumul_infections(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_daily_exposures'
    ax = ch_daily_exposures(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_hosp_admits'
    ax = ch_hosp_admits(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_daily_deaths'
    ax = ch_daily_deaths(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    thischart = 'ch_detection_rt'
    ax = ch_detection_rt(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    thischart = 'ch_postestshare'
    ax = ch_postestshare(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    thischart = 'ch_positivetests'
    ax = ch_positivetests(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    thischart = 'ch_totaltests'
    ax = ch_totaltests(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    thischart = 'ch_doubling_rt'
    ax = ch_doubling_rt(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web:
        plt.style.use(personalsitestyle); plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight'); plt.style.use(covidoutlookstyle); plt.savefig('../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], thischart), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    if pdf_out and type(pdf_out) == str: pdf_obj.close()

    print('Peak Hospitalization Date: ', df_agg.hospitalized.idxmax().strftime("%d %b, %Y"))
    print('Peak Hospitalization #: {:.0f}'.format(df_agg.hospitalized.max()))
    print('Peak ICU #: {:.0f}'.format(df_agg.icu.max()))
    print('Peak Ventilator #: {:.0f}'.format(df_agg.vent.max()))