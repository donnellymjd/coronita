#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os, time, stat, io
from scipy.stats import gamma, norm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')
import plotly.io as pio
from covid_data_helper import *
from coronita_chart_helper import *
from coronita_model_helper import *
from coronita_web_helper import *

from matplotlib.backends.backend_pdf import PdfPages


# # Data Ingestion

# ## Bring in State Data

# In[2]:


df_st_testing = get_covid19_tracking_data()


# In[3]:


df_st_testing.dateChecked


# In[4]:


df_census = get_census_pop()


# In[ ]:


df_counties = get_complete_county_data() #get_nyt_counties()

# In[ ]:


counties_geo = get_counties_geo()


# In[ ]:


df_jhu_counties = get_jhu_counties()


# In[ ]:


# df_counties_enhanced = pd.merge(
#     df_census[df_census.SUMLEV==50], 
#     df_counties.loc[[(df_counties.index.levels[0][-1])]].reset_index()[['dt','fips','cases','deaths']], 
#     on='fips', how='left')
# df_counties_enhanced['cases_per100k']= df_counties_enhanced['cases'].mul(1e5).div(df_counties_enhanced['pop2019'])
# df_counties_enhanced.head()


# In[ ]:


df_st_testing_fmt = df_st_testing.copy()
df_st_testing_fmt = df_st_testing_fmt.rename(columns={'death':'deaths','positive':'cases'}).unstack('code')


# In[ ]:

df_interventions = get_state_policy_events()


# In[ ]:


df_goog_mob_us = get_goog_mvmt_us()
df_goog_mob_us = df_goog_mob_us[df_goog_mob_us.state.isnull()].set_index('dt')


# In[ ]:


df_goog_mob_state = get_goog_mvmt_state()


# In[ ]:


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
covid_params['d_til_death'] =  17.0
covid_params['policy_trigger'] = True
covid_params['policy_trigger_once'] = True

# # Scenarios

# In[ ]:


df_rts_allregs = pd.DataFrame()
df_wavg_rt_conf_allregs = pd.DataFrame()

for region in df_st_testing_fmt['date'].columns:
    try:
        model_dict = make_model_dict_state(region, abbrev_us_state, df_census, df_st_testing_fmt, covid_params, 150)
        
        this_reg_df_rts = pd.DataFrame(model_dict['df_rts'].stack(), columns=[region])
        this_reg_df_wavg = pd.DataFrame(
            model_dict['df_rts_conf'].sort_index().unstack('metric')['weighted_average'].stack(), columns=[region])

        df_rts_allregs = pd.concat([df_rts_allregs, this_reg_df_rts], axis=1)
        df_wavg_rt_conf_allregs = pd.concat([df_wavg_rt_conf_allregs,this_reg_df_wavg], axis=1)
    except:
        print('Cannot forecast {}'.format(region))

df_rts_allregs.index.names = ['dt','metric']

fig = ch_rt_summary(df_wavg_rt_conf_allregs)
fig.write_html('./output/state_fore/rt_summary.html')
fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/rt_summary.html')


# In[ ]:


cover_file = './output/state_fore/coverpage.pdf'
chart_file = './output/state_fore/charts.pdf'

df_fore_allstates = pd.DataFrame()

try:
    df_prevfore_allstates = pd.read_pickle('./output/df_fore_allstates_{}.pkl'.format(
        (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y%m%d")))
except:
    if 'df_fore_allstates' in globals().keys():
        if df_fore_allstates.shape[0] > 0:
            df_prevfore_allstates = df_fore_allstates.copy()

allstate_model_dicts = {}
l_pdfs_out = []

for state in df_census.state.unique():
    print(state)
    
    model_dict = make_model_dict_state(state, abbrev_us_state, df_census, df_st_testing_fmt, covid_params, 150,
                                       df_mvmt=df_goog_mob_state
#                                      , df_interventions=df_interventions
                                      )
    
    fig = ch_statemap2(df_counties.query('dt == dt.max() and state == "{}"'.format(state)), 
                       abbrev_us_state[state], 
                       df_counties.query('dt == dt.max()').cases_per100k.quantile(.9),
                       counties_geo
                      )
    fig = add_plotly_footnote(fig)
    pio.orca.shutdown_server()
    fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/{}_casepercap_cnty_map.html'.format(
        model_dict['region_code']))
    try:
        pio.orca.shutdown_server()
        fig.write_image(cover_file, scale=2)
    except:
        pio.orca.shutdown_server()

    pdf_obj = PdfPages(chart_file)

    try:
        first_guess = df_prevfore_allstates[state].first_valid_index()[0]
    except:
        first_guess = pd.Timestamp('2020-02-17')

    df_agg, df_all_cohorts, model_dict = model_find_start(first_guess, model_dict)

    allstate_model_dicts[state] = model_dict
    df_fore_allstates = pd.concat([df_fore_allstates,pd.DataFrame(df_agg.stack(), columns=[state])], axis=1)

    run_all_charts(allstate_model_dicts[state], 
                   df_fore_allstates[state].unstack('metric'), 
                   r'No Change in Future $R_{t}$ Until 20% Hospital Capacity Trigger', pdf_obj, False, 
                   pub2web=True)
    pdf_obj.close()

    pdf_out = './output/state_fore/coronita_forecast_{}_{}.pdf'.format(
        state, pd.Timestamp.today().strftime("%Y%m%d"))
    gs_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile='
    cmd_str = '{0}{1} {2} {3}'.format(
        gs_cmd, pdf_out, cover_file, chart_file)
    os.system(cmd_str)
    l_pdfs_out.append(pdf_out)

pdf_out = './output/state_fore/coronita_forecast_{}_{}.pdf'.format(
    'us', pd.Timestamp.today().strftime("%Y%m%d"))
gs_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile='
cmd_str = '{0}{1} {2}'.format(
    gs_cmd, pdf_out, ' '.join(sorted(l_pdfs_out)))
os.system(cmd_str)


# In[ ]:


df_fore_allstates.unstack('metric').to_csv(
    './output/df_fore_allstates_{}.csv'.format(pd.Timestamp.today().strftime("%Y%m%d")),
    encoding='utf-8')
df_fore_allstates.to_pickle('./output/df_fore_allstates_{}.pkl'.format(pd.Timestamp.today().strftime("%Y%m%d")))

import pickle

asmd_filename = './output/allstate_model_dicts_{}.pkl'.format(pd.Timestamp.today().strftime("%Y%m%d"))

with open(asmd_filename, 'wb') as handle:
    pickle.dump(allstate_model_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


# ## US Charts

# In[ ]:


cover_file = './output/state_fore/coverpage.pdf'
chart_file = './output/state_fore/charts.pdf'
model_dict = make_model_dict_us(df_census, df_st_testing_fmt, covid_params, d_to_forecast = 75,
                               df_mvmt=df_goog_mob_us, df_interventions=df_interventions)

fig = ch_statemap2(df_counties.query('dt == dt.max()'), 
                   'USA', 
                   df_counties.query('dt == dt.max()').cases_per100k.quantile(.9),
                   counties_geo,
                   fitbounds=False)
# fig.show()

pio.orca.shutdown_server()
fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/{}_casepercap_cnty_map.html'.format(
    model_dict['region_code']))
fig.write_image(cover_file, scale=2)

pdf_obj = PdfPages(chart_file)

df_fore_us = df_fore_allstates.sum(axis=1, skipna=True).unstack('metric').dropna(how='all')
tot_pop = df_fore_us[['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered']].sum(axis=1)
max_tot_pop = tot_pop.max()
df_fore_us.loc[tot_pop<max_tot_pop, 'susceptible'] = df_fore_us['susceptible'] + (max_tot_pop - tot_pop)

run_all_charts(model_dict,
               df_fore_us,
               scenario_name=r'No Change in Future $R_{t}$ Until 20% Hospital Capacity Trigger',
               pdf_out=pdf_obj,
               show_charts=False,
               pub2web=True
              )
pdf_obj.close()

pdf_out = './output/state_fore/coronita_forecast_{}_{}.pdf'.format(
    'us', pd.Timestamp.today().strftime("%Y%m%d"))
gs_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile='
cmd_str = '{0}{1} {2}'.format(
    gs_cmd, pdf_out, ' '.join([cover_file, chart_file] + sorted(l_pdfs_out)))
os.system(cmd_str)


# In[ ]:


cmd_str = '{0}{1} {2}'.format(
    gs_cmd, pdf_out, ' '.join([cover_file, chart_file] + sorted(l_pdfs_out)))
os.system(cmd_str)


infectious_contact_prob = df_fore_allstates.loc[
    pd.Timestamp.today().date()].loc[['exposed','infectious']].sum().div(
    df_census[df_census.SUMLEV==40].set_index('state')['pop2019']).sort_values()

df_chart = (1-(1-infectious_contact_prob)**10).reset_index()
df_chart.columns = ['state','Exposure Probability (%)']
df_chart['Exposure Probability (%)'] = df_chart['Exposure Probability (%)'].mul(100).round(1)

import plotly.express as px


chart_title = 'US: Current Model-Estimated COVID-19 Exposure Probability Per 10 Contacts'

fig = px.choropleth(df_chart[['state','Exposure Probability (%)']],
                    locations=df_chart['state'],
                    locationmode="USA-states",
                    color='Exposure Probability (%)',
                    color_continuous_scale="BuPu",
                    title=chart_title,
                    projection='albers usa'
                    )

fig = add_plotly_footnote(fig)

fig.write_html('./output/state_fore/ch_exposure_prob.html')
fig.write_html('../donnellymjd.github.io/_covid19/datacenter/plotly/ch_exposure_prob.html')

# In[ ]:


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

l_charts = ['ch_rt_confid', 
            'ch_positivetests', 'ch_totaltests', 'ch_postestshare',
            'ch_detection_rt',
            'ch_statemap', 'ch_googmvmt',
            'ch_rts', 'ch_exposed_infectious', 'ch_hosp', 
            'ch_population_share',
            'ch_cumul_infections', 'ch_daily_exposures', 'ch_hosp_admits', 'ch_daily_deaths', 
            'ch_doubling_rt'
           ]

for state_code in list(df_census.state.unique())+['US']:
    l_content = ['<h3>How Fast is COVID-19 Currently Spreading?</h3>']
    
    for thischart in l_charts:
        if thischart == 'ch_statemap':
            l_content.append('{{% include_relative plotly/{}_casepercap_cnty_map.html %}}'.format(state_code))
        else:
            l_content.append("<img src='/assets/images/covid19/{}_{}.png' class='image fit'>".format(
                state_code, thischart))
            
        if thischart in dict_ch_defs.keys():
            l_content.append(dict_ch_defs[thischart]+'<br><br>')
    
    l_content.insert(2, '<!--more-->')
    
    l_content.insert(16, '<h3>Model and Forecast Results</h3>')
    
    if state_code == 'US':
        state_name = 'United States'
        final_md = state_md_template.format(state_name, state_code, '\n'.join(l_content), 1, 'fa-flag-usa')
    else:
        state_name = abbrev_us_state[state_code]
        final_md = state_md_template.format(state_name, state_code, '\n'.join(l_content), 3, 'fa-head-side-mask')

    filename = "../donnellymjd.github.io/_covid19/datacenter/{}.md".format(state_code)        
    
    with open(filename, "w") as file:
        file.write(final_md)


# In[ ]:


import subprocess
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


# In[ ]:




