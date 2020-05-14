import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.style.use('fivethirtyeight')


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
    df_chart = df_agg[['exposed', 'infectious']]
    df_chart = df_chart.clip(lower=0)

    ax = df_chart.plot.area(figsize=[14, 8], title='Simultaneous Infections Forecast\n'+chart_title,
                            legend=True, color=['#e5ae38', '#fc4f30'])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend(['Exposed Population', 'Infectious Population'],
               loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax2 = ax.twinx()

    r_t = model_dict['df_rts']['rt_joint_est'].copy()
    r_t = r_t.reindex(df_chart.index)
    r_t[df_chart.index[0]:df_chart.index[-1]].plot(ax=ax2, color='black', linewidth=2, linestyle='--',
                                                   label=r'Reproduction Factor ($R_t$) - Right Axis', legend=True)
    plt.legend(loc="lower right")
    ax.set_ylim([0, ax.axes.get_yticks().max()])
    ax.set_yticks(np.linspace(0, ax.axes.get_yticks().max(), 5))
    ax.set_ylabel('People')
    ax2.set_ylim([0, ax2.axes.get_yticks().max()])
    ax2.set_yticks(np.linspace(0, ax2.axes.get_yticks().max(), 5))
    ax2.set_ylabel('$R_t$ Estimate')
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})

    ref_line = r_t[df_chart.index[0]:df_chart.index[-1]].copy()
    ref_line.loc[:] = 1.0
    ref_line.plot(ax=ax2, color=['#008fd5'], legend=True, linewidth=1, label='Reference Line: $R_t = 1$')
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

    
    return


def ch_cumul_infections(df_agg, model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    df_chart = df_agg[['exposed', 'infectious', 'recovered', 'hospitalized', 'deaths']].sum(axis=1)
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
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    


def ch_daily_exposures(df_agg, model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    df_chart = df_agg['exposed_daily']

    ax = df_chart.plot(figsize=[14, 8], title='Daily Exposures Forecast\n'+chart_title,
                       legend=True, color=['#e5ae38'],
                       label='Forecast Daily New Infections (Exposed)')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if 'cases_tot' in model_dict['df_hist'].columns:
        model_dict['df_hist']['cases_tot'].loc[df_chart.index[0]:].diff().plot(
            ax=ax, linestyle=':', legend=True, color=['#e5ae38'],
            label='Reported Daily New Infections (Exposed)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')


def ch_hosp(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg[['hospitalized', 'icu', 'vent', 'deaths']].copy()
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
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    

def ch_hosp_admits(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg['hosp_admits']

    ax = df_chart.plot(figsize=[14, 8], title='Daily Hospital Admissions Forecast\n'+chart_title,
                       label='Forecast Hospital Admissions',
                       color='#6d904f')
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    if 'hosp_admits' in model_dict['df_hist'].columns:
        model_dict['df_hist']['hosp_admits'].plot(ax=ax, linestyle=':', legend=True,
                                                  label='Reported Hospital Admissions',
                                                  color='#6d904f'
                                                  )
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')

def ch_daily_deaths(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg['deaths'].diff()

    ax = df_chart.plot(figsize=[14, 8], title='Daily Deaths Forecast\n'+chart_title,
                       label='Forecast Daily Deaths',
                       color='#fc4f30')
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    if 'deaths_daily' in model_dict['df_hist'].columns:
        model_dict['df_hist']['deaths_daily'].plot(ax=ax, linestyle=':', legend=True,
                                                  label='Reported Daily Deaths',
                                                  color='#fc4f30'
                                                  )
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')


def ch_doubling_rt(df_agg, model_dict, param_str, chart_title=""):
    ## DOUBLING RATE CHART
    df_chart = np.log(2) / df_agg[['hospitalized', 'deaths']].pct_change()
    #     df_chart = df_chart.loc[hosp_obs_dt:]
    df_chart = df_chart.iloc[8:]

    ax = df_chart.plot(figsize=[14, 8], title='Doubling Rate Forecast\n' + chart_title,
                       color=['#008fd5', '#e5ae38'])
    if 'hosp_concur' in model_dict['df_hist'].columns:
        hosp_dr = np.log(2) / model_dict['df_hist']['hosp_concur'].pct_change().rolling(3).mean()
        hosp_dr.plot(ax=ax, linestyle=':', legend=True, color=['#008fd5'],
                     label='Reported Concurrent Hospitalizations')

    if 'deaths_tot' in model_dict['df_hist'].columns:
        deaths_dr = np.log(2) / model_dict['df_hist']['deaths_tot'].pct_change().rolling(3) \
                                    .mean()
        deaths_dr.plot(ax=ax, linestyle=':', legend=True, color=['#e5ae38'],
                       label='Reported Total Deaths')
    plt.yscale('log')
    _ = ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')


def ch_population_share(df_agg, model_dict, param_str, chart_title=""):
    df_chart = df_agg[['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered']]
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
    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)
    ax2 = ax.twinx()
    pd.Series(1.0, index=df_chart.index).plot(ax=ax2, color='black', linewidth=0, linestyle='--', legend=False)

    ax.set_ylim([0, df_chart.sum(axis=1).max()])
    ax.set_yticks(np.linspace(0, df_chart.sum(axis=1).max(), 5))
    ax.set_ylabel('Population')
    ax2.set_ylim([0, 1.0])
    ax2.set_yticks(np.linspace(0, 1.0, 5))
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    

def ch_rts(model_dict, param_str, chart_title=""):
    plt.style.use('fivethirtyeight')
    solo_rts = [x for x in model_dict['df_rts'].columns if x in
                ['rt_deaths_daily', 'rt_hosp_concur', 'rt_hosp_admits', 'rt_pos_test_share_daily', 'rt_cases_daily']]
    ax = model_dict['df_rts'][solo_rts].dropna(how='all').plot(figsize=[14, 8], alpha=0.2,
                                                               title=r'Reproduction Rate ($R_{t}$) Estimates'+'\n'+chart_title,
                                                               legend=True)

    model_dict['df_rts']['rt_joint_est'].dropna().plot(ax=ax, legend=True)

    ax.set_ylim([0,ax.get_ylim()[1]])

    plt.legend(loc='upper left', bbox_to_anchor=(1.07, 0.95), ncol=1)

    ax.text(1.08, 0.05, param_str, transform=ax.transAxes,
            verticalalignment='bottom', bbox={'ec': 'black', 'lw': 1})
    ax.set_xlabel('')
    footnote_str = 'Author: Michael Donnelly (twtr: @donnellymjd)\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b %Y"))
    plt.annotate(footnote_str,
                 (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
    

def run_all_charts(model_dict, df_agg, scenario_name='', pdf_out=False):
    chart_title = "{0}: {1} Scenario".format(
        model_dict['region_name'], scenario_name)

    param_str = param_str_maker(model_dict)

    if pdf_out and type(pdf_out) == str:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = PdfPages(pdf_out)
    elif pdf_out:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_obj = pdf_out

    ch_rts(model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_exposed_infectious(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_hosp(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_population_share(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_cumul_infections(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_daily_exposures(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_hosp_admits(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_daily_deaths(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    ch_doubling_rt(df_agg, model_dict, param_str, chart_title)
    if pdf_out: plt.tight_layout(); pdf_obj.savefig()
    plt.show()

    if pdf_out and type(pdf_out) == str: pdf_obj.close()

    print('Peak Hospitalization Date: ', df_agg.hospitalized.idxmax().strftime("%d %b, %Y"))
    print('Peak Hospitalization #: {:.0f}'.format(df_agg.hospitalized.max()))
    print('Peak ICU #: {:.0f}'.format(df_agg.icu.max()))
    print('Peak Ventilator #: {:.0f}'.format(df_agg.vent.max()))