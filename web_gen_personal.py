import pandas as pd
import numpy as np
import os, time, stat, io, glob, pickle, subprocess
from scipy.stats import gamma, norm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.io as pio

from covid_data_helper import *
from coronita_chart_helper import *
from coronita_web_helper import *
from coronita_bokeh_helper import *

from matplotlib.backends.backend_pdf import PdfPages

### Settings and Functions for Personal Website ###
# plt.style.use('./personal_web.mplstyle')
plt.style.use('fivethirtyeight')

def footnote_str_maker():
    footnote_str = 'Author: Michael Donnelly | twtr: @donnellymjd | www.michaeldonnel.ly\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b, %Y at %I:%M %p"))
    return footnote_str


def add_plotly_footnote(fig):
    fig.update_layout(
                  annotations=[
                      dict(x = 0, y = -0.06, font_size=10, showarrow=False,
                           xref='paper', yref='paper',
                           xanchor='left', yanchor='auto', xshift=0, yshift=0,
                           text='Author: Michael Donnelly | twtr: <a href="https://twitter.com/donnellymjd">@donnellymjd</a> | <a href="http://{0}">{0}</a>'.format(
                           'www.michaeldonnel.ly')
                          ),
                      dict(x = 0, y = -0.09, font_size=10, showarrow=False,
                           xref='paper', yref='paper',
                           xanchor='left', yanchor='auto', xshift=0, yshift=0,
                           text='Chart created on {}'.format(pd.Timestamp.today().strftime("%d %b %Y"))
                          )
                  ]
                 )
    return fig

state_md_template = '''---
title: {0}
layout: 'post'
statecode: {1}
order: {3}
icon: {4}
excerpt_separator: <!--more-->
level: 2
breadcrumb: {0}
---
{2}
'''


#####################################

######### DATA INGESTION ############

df_st_testing = get_covid19_tracking_data()

df_census = get_census_pop()

df_counties = get_complete_county_data()

counties_geo = get_counties_geo()

df_jhu_counties = get_jhu_counties()

df_st_testing_fmt = df_st_testing.copy()
df_st_testing_fmt = df_st_testing_fmt.rename(columns={'death':'deaths','positive':'cases'}).unstack('code')

df_interventions = get_state_policy_events()

df_goog_mob_us = get_goog_mvmt_us()
df_goog_mob_state = get_goog_mvmt_state(df_goog_mob_us)
df_goog_mob_us = df_goog_mob_us[df_goog_mob_us.state.isnull()].set_index('dt')

list_of_files = glob.glob('./output/df_fore_allstates_*.pkl') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
df_fore_allstates = pd.read_pickle(latest_file)

list_of_files = glob.glob('./output/allstate_model_dicts_*.pkl') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
with open(latest_file, 'rb') as handle:
    allstate_model_dicts = pickle.load(handle)

list_of_files = glob.glob('./output/df_wavg_rt_conf_allregs_*.pkl') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
df_wavg_rt_conf_allregs = pd.read_pickle(latest_file)

#####################################

fig = ch_rt_summary(df_wavg_rt_conf_allregs)
fig = add_plotly_footnote(fig)
fig.write_html('./output/state_fore/rt_summary.html', include_plotlyjs='cdn')
fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/rt_summary.html', include_plotlyjs='cdn')

fig = ch_exposure_prob(df_fore_allstates,
                       df_census[df_census.SUMLEV == 40].set_index('state')['pop2019'])
fig = add_plotly_footnote(fig)
fig.write_html('./output/state_fore/ch_exposure_prob.html', include_plotlyjs='cdn')
fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/ch_exposure_prob.html', include_plotlyjs='cdn')

cover_file = './output/state_fore/coverpage.pdf'
chart_file = './output/state_fore/charts.pdf'
l_pdfs_out = []

l_charts = ['ch_rt_confid',
           'ch_positivetests', 'ch_totaltests', 'ch_postestshare',
           'ch_detection_rt',
           'ch_statemap', 'ch_googmvmt',
           'ch_rts', 'ch_exposed_infectious', 'ch_hosp_concur','ch_deaths_tot',
           'ch_population_share',
           'ch_cumul_infections', 'ch_daily_exposures', 'ch_hosp_admits', 'ch_daily_deaths',
           'ch_doubling_rt'
           ]

d_chart_fns = {'ch_rt_confid': ch_rt_confid,
 'ch_positivetests': ch_positivetests,
 'ch_totaltests': ch_totaltests,
 'ch_postestshare': ch_postestshare,
 'ch_detection_rt': ch_detection_rt,
 'ch_googmvmt': ch_googmvmt,
 'ch_rts': ch_rts,
 'ch_exposed_infectious': ch_exposed_infectious,
 'ch_hosp_concur': ch_hosp_concur,
 'ch_deaths_tot': ch_deaths_tot,
 'ch_population_share': ch_population_share,
 'ch_cumul_infections': ch_cumul_infections,
 'ch_daily_exposures': ch_daily_exposures,
 'ch_hosp_admits': ch_hosp_admits,
 'ch_daily_deaths': ch_daily_deaths,
 'ch_doubling_rt': ch_doubling_rt}

for state_code in list(df_census.state.unique()) + ['US']:
    print(state_code)
    model_dict = allstate_model_dicts[state_code]
    model_dict['footnote_str'] = footnote_str_maker()

    # fig = ch_statemap2(df_counties.query('dt == dt.max() and state == "{}"'.format(state_code)),
    #                    model_dict['region_name'],
    #                    df_counties.query('dt == dt.max()').cases_per100k.quantile(.9),
    #                    counties_geo
    #                   )
    fig = ch_statemap_casechange(model_dict, df_counties, counties_geo)
    fig = add_plotly_footnote(fig)
    pio.orca.shutdown_server()
    fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/{}_casepercap_cnty_map.html'.format(
        model_dict['region_code']), include_plotlyjs='cdn')

    try:
        pio.orca.shutdown_server()
        fig.write_image(cover_file, scale=2)
    except:
        pio.orca.shutdown_server()

    pdf_obj = PdfPages(chart_file)

    for ch_name, ch_fn in d_chart_fns.items():
        try:
            ax = ch_fn(model_dict)
            pdf_obj.savefig(bbox_inches='tight', pad_inches=1, optimize=True, facecolor='white')
            filename = '../donnellymjd.github.io/assets/images/covid19/{}_{}.png'.format(
                model_dict['region_code'], ch_name)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            # os.system('optipng {} &'.format(filename))
        except:
            print('Couldn\'t create {} {} chart.'.format(model_dict['region_code'], ch_name))

    pdf_obj.close()
    pdf_out = './output/state_fore/coronita_forecast_{}_{}.pdf'.format(
        state_code, pd.Timestamp.today().strftime("%Y%m%d"))
    gs_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile='
    cmd_str = '{0}{1} {2} {3}'.format(
        gs_cmd, pdf_out, cover_file, chart_file)
    os.system(cmd_str)
    l_pdfs_out.append(pdf_out)

    l_content = ['<h3>How Fast is COVID-19 Currently Spreading?</h3>']

    for thischart in l_charts:
        if thischart == 'ch_statemap':
            l_content.append('{{% include_relative plotly/{}_casepercap_cnty_map.html %}}'.format(state_code))
        else:
            l_content.append("<img src='/assets/images/covid19/{}_{}.png' class='image fit'>".format(
                state_code, thischart))

        if thischart in dict_ch_defs.keys():
            l_content.append(dict_ch_defs[thischart] + '<br><br>')

    l_content.insert(2, '<!--more-->')

    l_content.insert(16, '<h3>Model and Forecast Results</h3>')

    if state_code == 'US':
        pagerank = 1
        menu_icon = 'fa-flag-usa'
    else:
        pagerank = 3
        menu_icon = 'fa-head-side-mask'

    final_md = state_md_template.format(
        model_dict['region_name'], state_code, '\n'.join(l_content), pagerank, menu_icon)
    # co_final_md = co_state_md_template.format(model_dict['region_name'], state_code, '\n'.join(l_content))

    filename = "../donnellymjd.github.io/_covid19/datacenter/{}.md".format(state_code)
    # co_filename = "../COVIDoutlook/{}.md".format(state_code)

    with open(filename, "w") as file:
        file.write(final_md)
    # with open(co_filename, "w") as file:
    #     file.write(co_final_md)

pdf_out = './output/state_fore/coronita_forecast_{}_{}.pdf'.format(
    'us', pd.Timestamp.today().strftime("%Y%m%d"))
gs_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile='
cmd_str = '{0}{1} {2}'.format(
    gs_cmd, pdf_out, ' '.join(sorted(l_pdfs_out)))
os.system(cmd_str)

git_dir = '/Users/mdonnelly/repos/donnellymjd.github.io/'
git_commit_cmd = 'git commit -am "Auto update on {}"'.format(
    pd.Timestamp.today().strftime("%Y-%m-%d at %I:%M %p"))
print(git_commit_cmd)
status_out = subprocess.check_output('git status',
                                     cwd=git_dir, shell=True).decode()
print(status_out)
commit_out = subprocess.check_output(git_commit_cmd,
                                     cwd=git_dir, shell=True).decode()
print(commit_out)
push_out = subprocess.check_output('git push',
                                   cwd=git_dir, shell=True).decode()
print(push_out)

os.system('say -v "Victoria" "Your personal website is ready."')