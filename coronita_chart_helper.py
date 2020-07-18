import pandas as pd
import numpy as np
from collections import OrderedDict
import io

from coronita_model_helper import *

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('fivethirtyeight')


def footnote_str_maker():
    footnote_str = 'Author: Michael Donnelly | twtr: @donnellymjd | www.michaeldonnel.ly\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b, %Y at %I:%M %p"))
    return footnote_str

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
                       bar2_series = None, bar2_name='', bar2_color='#e5ae38'):
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

    
    plt.annotate(footnote_str_maker(),
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

def ch_exposed_infectious(df_agg, model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    df_chart = df_agg[['exposed', 'infectious']].dropna(how='all')
    df_chart = df_chart.clip(lower=0)

    ax = df_chart.plot.area(figsize=[14, 8], title='Simultaneous Infections Forecast\n'+chart_title,
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
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_cumul_infections(df_agg, model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    df_chart = df_agg[['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths']].sum(axis=1).dropna(how='all')
    df_chart = df_chart.clip(lower=0)
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot(figsize=[14, 8], title='Cumulative Infections Forecast\n'+chart_title,
                       legend=True, label='Forecast Cumulative Infections',
                       color=['black'])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if 'cases_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['cases_tot'].loc[df_chart.index[0]:].plot(
            ax=ax, linestyle=':', legend=True, color=['black'],
            label='Reported Cumulative Infections')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_daily_exposures(df_agg, model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    df_chart = df_agg['exposed_daily'].dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title='Daily Exposures Forecast\n'+chart_title,
                       legend=True, color=['#e5ae38'],
                       label='Forecast Daily New Infections (Exposed)')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if 'cases_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['cases_tot'].loc[df_chart.index[0]:].diff().plot(
            ax=ax, linestyle=':', legend=True, color=['#e5ae38'],
            label='Reported Daily New Infections (Exposed)')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_hosp(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg[['hospitalized', 'icu', 'vent', 'deaths']].dropna(how='all').copy()
    df_chart = df_chart.rename(columns={'hospitalized':'Forecast Concurrent Hospitalizations',
                                        'icu':'Forecast ICU Cases',
                                        'vent':'Forecast Ventilations',
                                        'deaths':'Forecast Cumulative Deaths'})

    ax = df_chart.plot(figsize=[14, 8], title='Hospitalization and Deaths Forecast\n'+chart_title,
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
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax
    

def ch_hosp_admits(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg['hosp_admits'].dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title='Daily Hospital Admissions Forecast\n'+chart_title,
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
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_daily_deaths(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg['deaths'].diff().dropna(how='all')

    ax = df_chart.plot(figsize=[14, 8], title='Daily Deaths Forecast\n'+chart_title,
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
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_doubling_rt(df_agg, model_dict, param_str, chart_title=""):
    ## DOUBLING RATE CHART
    df_chart = np.log(2) / df_agg[['exposed_daily', 'hospitalized', 'deaths']].pct_change().rolling(7, center=True).mean().dropna(how='all')
    df_chart = df_chart.mask(df_chart < 0)
    df_chart = df_chart.rename(columns={'exposed_daily':'New Cases',
                                        'hospitalized':'Hospitalizations',
                                        'deaths':'Daily Deaths'})
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot(figsize=[14, 8], title='Doubling Rate Forecast\n' + chart_title,
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
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax


def ch_population_share(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg[['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered']].dropna(how='all')
    df_chart = df_chart.rename(columns={'susceptible':'Forecast Susceptible Population',
                                        'exposed':'Forecast Exposures',
                                        'infectious':'Forecast Infectious',
                                        'hospitalized':'Forecast Hospitalizations',
                                        'recovered':'Forecast Recoveries',
                                        'deaths':'Forecast Deaths'})
    df_chart = df_chart.clip(lower=0)
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot.area(figsize=[14, 8], title='Population Overview Forecast\n'+chart_title)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax2 = ax.twinx()
    pd.Series(1.0, index=df_chart.index).plot(ax=ax2, color='black', linewidth=0, linestyle='--', legend=False)

    ax.set_ylim([0, df_chart.sum(axis=1).max()])
    ax.set_yticks(np.linspace(0, df_chart.sum(axis=1).max(), 5))
    ax.set_ylabel('Population')
    ax2.set_ylim([0, 1.0])
    ax2.set_yticks(np.linspace(0, 1.0, 5), minor=False)
    ax2.set_yticks(np.linspace(0, 1.0, 25), minor=True)
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax2.tick_params(axis='y', reset=True, direction='inout', which='minor', color='black', left=True,
                    length=10 , width=1)
    plt.box(on=None)
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_rts(model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    # solo_rts = [x for x in model_dict['df_rts'].columns if x in
    #             ['rt_deaths_daily', 'rt_hosp_concur', 'rt_hosp_admits', 'rt_pos_test_share_daily', 'rt_cases_daily']]

    df_just_rts = model_dict['df_rts'].dropna(how='all')

    solo_rts = [x for x in df_just_rts.columns if x in
                ['test_share', 'deaths_daily', 'hosp_concur', 'hosp_admits',
                 'cases_daily'] ]
    ax = df_just_rts[solo_rts].dropna(how='all').plot(figsize=[14, 8], alpha=0.2,
                                                               title=r'Reproduction Rate ($R_{t}$) Estimates'+'\n'+chart_title,
                                                               legend=True)

    df_just_rts['weighted_average'].dropna().plot(ax=ax, legend=True)

    ax.set_ylim([0,ax.get_ylim()[1]])

    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1, title='Reproduction Factor Estimates')

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_rt_confid(df_rt, param_str, chart_title=""):
    rt_name = df_rt.columns.levels[0][0]
    df_rt = df_rt[rt_name].dropna(how='all')

    # lower68, upper68, lower95, upper95 = df_rt.iloc[:,1:5]
    ax = df_rt['rt'].dropna(how='all').plot(figsize=[14, 8],
                                     title=r'Reproduction Rate ($R_{t}$) Estimate'+'\n'+chart_title,
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
    
    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    return ax

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

    fig = px.choropleth(df_chart.reset_index()[['fips','cases_per100k']].dropna(),
                        geojson=counties,
                        locations='fips', color='cases_per100k',
                        color_continuous_scale="amp",
                        range_color=(0, max((scale_max//200),1)*200),
                        scope="usa",
                        title=chart_title,
                        projection='albers usa'
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

def ch_statemap_casechange(df_chart, region_name, scale_max, counties, fitbounds='locations'):
    import plotly.express as px

    chart_title = region_name + ': COVID-19 Cases Per 100k Residents'

    fig = px.choropleth(df_chart.reset_index()[['fips','cases_per100k']].dropna(),
                        geojson=counties,
                        locations='fips', color='cases_per100k',
                        color_continuous_scale="amp",
                        range_color=(0, max((scale_max//200),1)*200),
                        scope="usa",
                        title=chart_title,
                        projection='albers usa'
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

def ch_totaltests(model_dict):
    ax = bar_and_line_chart(bar_series=model_dict['df_hist']['pos_neg_tests_daily'].dropna(how='all'),
                       bar_name='# of Negative Tests',
                       line_series=model_dict['df_hist']['pos_neg_tests_daily'].rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average', yformat='{:0,.0f}',
                       chart_title='{}: Total COVID-19 Tests Per Day'.format(model_dict['region_name']),
                       bar2_series=model_dict['df_hist']['cases_daily'], bar2_name='# of Positive Tests'
                       )
    return ax

def bk_totaltests(model_dict):
    df_chart = model_dict['df_hist'][['cases_daily', 'pos_neg_tests_daily']].clip(lower=0)
    df_chart['neg_tests_daily'] = (df_chart['pos_neg_tests_daily'] - df_chart['cases_daily']).clip(lower=0)

    p = bk_bar_and_line_chart(bar_series=df_chart['neg_tests_daily'].dropna(how='all'),
                       bar_name='# of Negative Tests',
                       line_series=df_chart[['cases_daily','pos_neg_tests_daily']].max(axis=1).rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average Total Tests', yformat='{:0,.0f}',
                       chart_title='{}: Total COVID-19 Tests Per Day'.format(model_dict['region_name']),
                       bar2_series=df_chart['cases_daily'], bar2_name='# of Positive Tests'
                       )
    return p

def ch_positivetests(model_dict):
    ax = bar_and_line_chart(bar_series=model_dict['df_hist']['cases_daily'].dropna(how='all'),
                       bar_name='# of Positive Tests', bar_color='#e5ae38',
                       line_series=model_dict['df_hist']['cases_daily'].rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average', yformat='{:0,.0f}',
                       chart_title='{}: Positive COVID-19 Tests Per Day'.format(model_dict['region_name'])
                       )
    return ax

def ch_postestshare(model_dict):
    df_chart = model_dict['df_hist'][['cases_daily', 'pos_neg_tests_daily']].clip(lower=0)
    df_chart = df_chart.div(df_chart['pos_neg_tests_daily'], axis=0).dropna(how='all')

    ax = df_chart.plot(kind='area', stacked=True,
                       title='{}: COVID-19 Positivity Rate'.format(model_dict['region_name']),
                       figsize=[14, 8],
                       color=['#e5ae38', '#008fd5'])
    df_chart['sevendayavg'] = df_chart['cases_daily'].mask(df_chart['cases_daily']>=1.0
                                                           ).rolling(7, min_periods=1).mean()
    df_chart['sevendayavg'].plot(ax=ax, linestyle='-', color='#fc4f30')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    plt.legend(title='Legend', bbox_to_anchor=(1.07, 0.95), ncol=1,
               labels=['% Positive Tests', '% Negative Tests', '7-Day Rolling Average - Positivity Rate'])
    ax.set_ylim(0, min(1, df_chart['sevendayavg'].max() * 1.25))
    ax.set_xlabel('')

    plt.annotate(footnote_str_maker(),
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

    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    return ax

def ch_detection_rt(df_agg, model_dict, param_str, chart_title=""):
    df_chart = model_dict['df_hist']['cases_daily'].rolling(7).mean().div(df_agg['exposed_daily']).dropna()

    ax = df_chart.plot(
        figsize=[14, 8],
        title='COVID-19 Daily Infection Detection Rate\n' + chart_title,
        legend=False
    )
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')

    plt.annotate(footnote_str_maker(),
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    return ax



def run_all_charts(model_dict, df_agg, scenario_name='', pdf_out=False, show_charts=True, pub2web=False):
    chart_title = "{0}: {1} Scenario".format(
        model_dict['region_name'], scenario_name)

    param_str = param_str_maker(model_dict)
    event_lines = model_dict['df_interventions']

    if pdf_out and type(pdf_out) == str:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = PdfPages(pdf_out)
    elif pdf_out:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = pdf_out

    ax = ch_rt_confid(model_dict['df_rts_conf'][['weighted_average']].unstack('metric'), param_str, model_dict['region_name'])
    if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_rt_confid'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_rts(model_dict, param_str, chart_title)
    if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_rts'), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    if model_dict['df_mvmt'].shape[0] > 0:
        ax = ch_googmvmt(model_dict)
        if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
        if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
        if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_googmvmt'), bbox_inches='tight')
        if show_charts:
            plt.show()
        else:
            plt.close()

    ax = ch_exposed_infectious(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_exposed_infectious'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_hosp(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_hosp'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()
    plt.close()

    ax = ch_population_share(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_population_share'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_cumul_infections(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_cumul_infections'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_daily_exposures(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_daily_exposures'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_hosp_admits(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_hosp_admits'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_daily_deaths(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_daily_deaths'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    ax = ch_detection_rt(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_detection_rt'), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    ax = ch_postestshare(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_postestshare'), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    ax = ch_positivetests(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_positivetests'), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    ax = ch_totaltests(model_dict)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_totaltests'), bbox_inches='tight')
    if show_charts:
        plt.show()
    else:
        plt.close()

    ax = ch_doubling_rt(df_agg, model_dict, param_str, chart_title)
    # if event_lines.shape[0] > 0: add_event_lines(ax, event_lines)
    if pdf_out: pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
    if pub2web: plt.savefig('../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(model_dict['region_code'], 'ch_doubling_rt'), bbox_inches='tight')
    if show_charts: 
        plt.show()
    else: 
        plt.close()

    if pdf_out and type(pdf_out) == str: pdf_obj.close()

    print('Peak Hospitalization Date: ', df_agg.hospitalized.idxmax().strftime("%d %b, %Y"))
    print('Peak Hospitalization #: {:.0f}'.format(df_agg.hospitalized.max()))
    print('Peak ICU #: {:.0f}'.format(df_agg.icu.max()))
    print('Peak Ventilator #: {:.0f}'.format(df_agg.vent.max()))