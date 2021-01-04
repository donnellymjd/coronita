import pandas as pd
import numpy as np
import os, time, stat, io, glob, pickle, subprocess
from scipy.stats import gamma, norm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib as mpl

from covid_data_helper import *
from coronita_chart_helper import *
from coronita_web_helper import *
from coronita_bokeh_helper import *

### Settings and Functions for Personal Website ###
# plt.style.use('file://Users/mdonnelly/repos/coronita/personal_covidoutlook.mplstyle')
plt.style.use('ggplot')

def footnote_str_maker():
    footnote_str = 'www.COVIDoutlook.info | twtr: @COVIDoutlook\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b, %Y at %I:%M %p"))
    return footnote_str


def add_plotly_footnote(fig):
    fig.update_layout(
                  annotations=[
                      dict(x = 0, y = 0,
                           xref='paper', yref='paper', font_size=12, showarrow=False,
                           xanchor='left', yanchor='top', xshift=0, yshift=0,
                            text='<a href="http://{0}">{0}</a> | twtr: <a href="https://twitter.com/COVIDoutlook">@COVIDoutlook</a>'.format(
                           'www.COVIDoutlook.info')
                          ),
                      dict(x = 0, y = -0.05,
                           xref='paper', yref='paper', font_size=10, showarrow=False,
                           xanchor='left', yanchor='top', xshift=0, yshift=0,
                           text='Chart created on {}'.format(pd.Timestamp.today().strftime("%d %b %Y"))
                          )
                  ]
                 )
    fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/donnellymjd/COVIDoutlook/master/assets/img/logo-whiteonblack.png", #"https://raw.githubusercontent.com/cldougl/plot_images/add_r_img/vox.png",
        xref="paper", yref="paper",
        x=0, y=0,
        sizex=0.15, sizey=0.15,
        xanchor="left", yanchor="bottom",
        layer='above'
    )
)
    return fig

state_md_template = '''---
title: {0}
layout: statereport
statecode: {1}
---
## {0}
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

df_hhs_hosp = get_hhs_hosp()

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


#### CREATE ONE OFF CHARTS (NATIONAL CHARTS) ####
# fig = ch_rt_summary(df_wavg_rt_conf_allregs)
# fig = add_plotly_footnote(fig)
# fig.write_html('../COVIDoutlook/plotly/rt_summary.html')

fig = ch_exposure_prob_anim(df_fore_allstates, df_census)
fig = add_plotly_footnote(fig)
fig.write_html('../COVIDoutlook/forecasts/plotly/ch_exposure_prob.html', include_plotlyjs='cdn')
fig.write_image('../COVIDoutlook/assets/images/covid19/ch_exposure_prob.png')

tab_html, df_tab, df_tab_us = tab_summary(df_st_testing_fmt, df_fore_allstates, df_census, df_wavg_rt_conf_allregs, df_hhs_hosp)
text_file = open("../COVIDoutlook/forecasts/plotly/summ_tab.html", "w")
text_file.write(tab_html)
text_file.close()
df_tab.to_csv('../COVIDoutlook/download/state_data_summary_tab.csv', encoding='utf-8')

## Compare Exposures ##
layout = bk_compare_exposures(df_census, df_fore_allstates)
curdoc().theme = bk_theme
script_loc = "/assets/js/compare.js"
js, div = components(layout)
js = js[37:-9]
with io.open('../COVIDoutlook' + script_loc, mode='w', encoding='utf-8') as f:
    f.write(js)

script = '<script src="{}" async="True"></script>'.format(script_loc)

resources = CDN.render()

template = Template('''---
layout: page
title: Compare States
banner: duotone2.png
---
{{ resources }}
{{ script }}
{{ div }}
''')

resources = CDN.render()

html = template.render(resources=resources,
                       script=script,
                       div=div)

with io.open('../COVIDoutlook/compare.md', mode='w', encoding='utf-8') as f:
    f.write(html)
####


## US Overview Page ##
region_code = 'US'
model_dict = allstate_model_dicts[region_code]

df_agg = model_dict['df_agg']
scenario_name = 'No Change in Future $R_{t}$ Until Reaching Hospital Capacity Triggers Lockdown'
chart_title = ""  # "{1} Scenario".format(model_dict['region_name'], scenario_name)
param_str = param_str_maker(model_dict)

reset_output()

p_cases = bk_positivetests(model_dict)
# p_cases.x_range = Range1d(pd.Timestamp('2020-03-01'),
#                           model_dict['df_hist'].last_valid_index())
p_cases = bk_overview_layout(p_cases, 2)

p_tests = bk_totaltests(model_dict)
# p_tests.x_range = p_cases.x_range
p_tests = bk_overview_layout(p_tests, 2)

p_positivity = bk_postestshare(model_dict)
# p_positivity.x_range = p_cases.x_range
p_positivity = bk_overview_layout(p_positivity)

p_rt_conf = bk_rt_confid(model_dict, simplify=False)
# p_rt_conf.x_range = p_cases.x_range
p_rt_conf = bk_overview_layout(p_rt_conf)

p_googmvmt = bk_googmvmt(model_dict)
# p_googmvmt.x_range = p_cases.x_range
p_googmvmt = bk_overview_layout(p_googmvmt)

p_det_rt = bk_detection_rt(df_agg, model_dict)
# p_det_rt.x_range = p_cases.x_range
p_det_rt = bk_overview_layout(p_det_rt)

p_pop_share = bk_population_share(model_dict)
# p_pop_share.x_range = p_cases.x_range
p_pop_share = bk_overview_layout(p_pop_share)

curdoc().theme = bk_theme
r1 = [p_cases, p_tests, p_positivity, p_rt_conf, p_googmvmt,
      p_det_rt, p_pop_share]

script_loc = "/assets/js/us_overview.js"
js, div = components(r1)
js = js[37:-9]
with io.open('../COVIDoutlook' + script_loc, mode='w', encoding='utf-8') as f:
    f.write(js)

script = '<script src="{}" async="True"></script>'.format(script_loc)

resources = CDN.render()

template = Template('''---
layout: page
title: Home - US Overview
banner: duotone-us.png
image: https://www.covidoutlook.info/assets/images/covid19/ch_exposure_prob.png
---
{{ resources }}
{{ script }}

<div class="w3-row-padding">
    <div class="w3-half">
        {{ div[0] }}
    </div>

    <div class="w3-half">
        {{ div[1] }}
    </div>
</div>
<hr>
<div class="w3-container">
    {{ div[2] }}
</div>
<hr>
<div class="w3-container">
    <iframe
    src="forecasts/plotly/ch_exposure_prob.html"
    style="margin:0; width:100%; height:500px; border:none;" scrolling="auto" sandbox="allow-scripts"
    ></iframe>
</div>
<hr>
<div>
    <h3> State Data Summary Table </h3>
    <iframe
    src="forecasts/plotly/summ_tab.html" scrolling="auto"
    style="margin:0; width:100%; height:500px; border:none;overflow-x:hidden;overflow-y:scroll;"
    sandbox="allow-scripts allow-top-navigation-by-user-activation"
    ></iframe>
</div>
<hr>
<div>
    <iframe
    src="forecasts/plotly/US_casepercap_cnty_map.html"
    style="margin:0; width:100%; height:800px; border:none;" scrolling="auto" sandbox="allow-scripts"
    ></iframe>
</div>
{% for chart in div[3:] %}
    <hr>
    <div class="w3-container">
        {{ chart }}
    </div>  
{% endfor %}
''')

resources = CDN.render()
exposure_prob = '{% include_relative forecasts/plotly/ch_exposure_prob.html %}'
case_change = '{% include_relative forecasts/plotly/US_casepercap_cnty_map.html %}'

html = template.render(resources=resources,
                       script=script,
                       div=div,
                       exposure_prob=exposure_prob,
                       case_change=case_change
                       )

with io.open('../COVIDoutlook/index.html', mode='w', encoding='utf-8') as f:
    f.write(html)
####

## Reproduction Rate Page ##
reset_output()
df_rts_allregs = pd.DataFrame()
df_wavg_rt_conf_allregs = pd.DataFrame()
l_rt_conf = []

model_dict = allstate_model_dicts['US']
l_rt_conf.append(bk_rt_confid(model_dict, True))
l_rt_conf[-1] = bk_overview_layout(l_rt_conf[-1], 1)

l_state_names = sorted([abbrev_us_state[code] for code in df_census.state.unique()])

for state_name in l_state_names:
    state_code = us_state_abbrev[state_name]
    model_dict = allstate_model_dicts[state_code]
    l_rt_conf.append(bk_rt_confid(model_dict, simplify=True))
    l_rt_conf[-1] = bk_repro_layout(l_rt_conf[-1], 2)

curdoc().theme = bk_theme
script_loc = "/assets/js/rts.js"
js, div = components(l_rt_conf)
js = js[37:-9]
with io.open('../COVIDoutlook' + script_loc, mode='w', encoding='utf-8') as f:
    f.write(js)

script = '<script src="{}" async="True"></script>'.format(script_loc)

resources = CDN.render()

template = Template('''---
layout: page
title: Reproduction Rates
banner: duotone4.png
---
{{ resources }}
{{ script }}

<div class="w3-row-padding">
    <div class="w3-container">
        {{ div[0] }}
    </div>
{% for chart in div[1:] %}
    <div class="w3-third">
        {{ chart }}
    </div>
{% endfor %}
</div>
''')

resources = CDN.render()

html = template.render(resources=resources,
                       script=script,
                       div=div)

with io.open('../COVIDoutlook/reproduction.md', mode='w', encoding='utf-8') as f:
    f.write(html)

##################################################


#### CREATE STATE CHARTS AND MD PAGES ####

l_charts = ['ch_positivetests', 'ch_totaltests', 'ch_postestshare','ch_rt_confid', 'ch_detection_rt',
           'ch_statemap', 'ch_googmvmt',
           'ch_exposed_infectious', 'ch_hosp_concur','ch_deaths_tot',
           'ch_population_share',
           'ch_cumul_infections', 'ch_daily_exposures', 'ch_hosp_admits', 'ch_daily_deaths'
           ]

d_chart_fns = {'ch_rt_confid': ch_rt_confid,
 'ch_positivetests': ch_positivetests,
 'ch_totaltests': ch_totaltests,
 'ch_postestshare': ch_postestshare,
 'ch_detection_rt': ch_detection_rt,
 'ch_googmvmt': ch_googmvmt,
 'ch_exposed_infectious': ch_exposed_infectious,
 'ch_hosp_concur': ch_hosp_concur,
 'ch_deaths_tot': ch_deaths_tot,
 'ch_population_share': ch_population_share,
 'ch_cumul_infections': ch_cumul_infections,
 'ch_daily_exposures': ch_daily_exposures,
 'ch_hosp_admits': ch_hosp_admits,
 'ch_daily_deaths': ch_daily_deaths}

state_plotly_html = '''<div>
    <iframe 
    src="/forecasts/plotly/{}_casepercap_cnty_map.html"
    style="margin:0; width:100%; height:800px; border:none; overflow:hidden;" scrolling="no"></iframe>
</div>'''

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
    fig.write_html('../COVIDoutlook/forecasts/plotly/{}_casepercap_cnty_map.html'.format(
        model_dict['region_code']), include_plotlyjs='cdn')

    for ch_name, ch_fn in d_chart_fns.items():
        try:
            ax = ch_fn(model_dict)
            filename = '../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(
                model_dict['region_code'], ch_name)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            # os.system('optipng {} &'.format(filename))
        except:
            print('Couldn\'t create {} {} chart.'.format(model_dict['region_code'], ch_name))

    statetab_output_cols = ['Riskiest State Rank', 'Population',
                   'Model Est\'d Active Infections', 'Current Reproduction Rate (Rt)',
                   'Days to Hospital Capacity',
                   'Total Cases', '14-Day Avg Daily Cases',
                   'Positivity Rate',
                   'Total Deaths', '14-Day Avg Daily Deaths',
                   'Hospitalized', '14-Day Avg Daily Hosp Admits'
                   ]
    if state_code == 'US':
        statetab = df_tab_us[statetab_output_cols[1:]].replace('nan', 'Not Available')
    else:
        statetab = df_tab.loc[df_tab.state == state_code, statetab_output_cols].replace('nan', 'Not Available')

    statetab_html = statetab.to_html(index=False, border=0, justify='center', escape=False)

    statetab_html = statetab_html.replace('▼', '<span style="color: green">▼</span>') \
        .replace('▲', '<span style="color: red">▲</span>') \
        .replace('▶', '<span style="color: #ffcc00">▶</span>')
    statetab_html = statetab_html.replace(
        'class="dataframe"', 'class="w3-table w3-striped w3-bordered w3-hoverable w3-medium"')
    statetab_html = statetab_html.replace(
        '<tr style="text-align: center;">', '<tr style="text-align: center;" class="w3-light-grey">')

    l_content = [statetab_html, '### How Fast is COVID-19 Currently Spreading?']

    for thischart in l_charts:
        if thischart == 'ch_statemap':
            # l_content.append('{{% include_relative plotly/{}_casepercap_cnty_map.html %}}'.format(state_code))
            l_content.append(state_plotly_html.format(state_code))
        else:
            l_content.append("<img src='/assets/images/covid19/{}_{}.png'>".format(
                state_code, thischart))

        if thischart in dict_ch_defs.keys():
            l_content.append(dict_ch_defs[thischart]+'\n- - - -')

    l_content.insert(15, '### Model and Forecast Results')

    final_md = state_md_template.format(model_dict['region_name'], state_code, '\n'.join(l_content))

    if state_code == 'US':
        filename = "../COVIDoutlook/forecasts/index.md".format(state_code)
    else:
        filename = "../COVIDoutlook/forecasts/{}.md".format(state_code)

    with open(filename, "w") as file:
        file.write(final_md)
#####################################


#### POST FORECAST DATA TO COVIDOUTLOOK ####

download_header = """---
layout: page
title: Data
banner: duotone5.png
---
### Data Sources
Our models and charts rely on data from a variety of publicly available datasets. These data providers depend on reliable and accurate reporting by national, state, and local governments. In many cases, the data reported by these governmental entities changes over time. This makes our jobs much more difficult as we have to account for some jurisdictions providing all necessary data on a daily basis, while other jurisdictions report less frequently or do not report some data at all. More and better data is necessary for high quality analysis and forecasts. If your state is showing incomplete data in our charts, ask your representatives to demand better reporting. 
 - [COVID-19 Tracking Project](https://api.covidtracking.com/v1/states/daily.csv) - The single most important source of data for our models. This volunteer run project collects and distributes state-level data on cases, testing, hospitalizations, and deaths. 
 - [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series) - County-level data on cases and deaths.
 - [New York Times](https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv) - County-level data on cases and deaths.
 - [NYC Dept of Health](https://github.com/nychealth/coronavirus-data/raw/master/case-hosp-death.csv) - Borough-level data on cases,  hospitalizations, and deaths.
 - [NYS Dept of Health](https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD) - County-level data on cases and total tests.
 - [US Census](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv) - County-level data on populations.
 - [Google Mobility Data](https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv) - While currently not used in our models, this data powers our mobility tracking charts.
 - [US Dept of HHS - Hopsital Capacity](https://healthdata.gov/api/3/action/package_show?id=060e4acc-241d-4d19-a929-f5f7b653c648) - After the CDC stopped collecting this data earlier in 2020, the HHS began collecting and hosting data reported by states of currently reported hospital bed capacity.

### Reference Downloads (comma separated values)
 - [COVID-19 Related Policy Actions by State - Source: KFF.org](https://raw.githubusercontent.com/donnellymjd/COVIDoutlook/master/download/df_interventions.csv)

### Our Forecasts (comma separated values)
{}
"""

list_of_files = glob.glob('../COVIDoutlook/download/df_fore_allstates_*.csv')
list_of_files = sorted(list_of_files)

file_dict = {}

for filename in list_of_files:
    this_date = pd.to_datetime(filename[43:51])
    file_dict[this_date] = (filename, this_date.strftime("%B %d, %Y"))

output_md = []

for key in reversed(sorted(file_dict.keys())):
    this_file = file_dict[key]
    output_md.append(' - [Forecast published on {0}](https://raw.githubusercontent.com/donnellymjd/COVIDoutlook/master{1})'.format(this_file[1], this_file[0][15:]))

final_md = download_header.format('\n'.join(output_md))

with open('../COVIDoutlook/data.md', "w") as file:
    file.write(final_md)

#####################################


#### COMMIT AND PUSH TO GITHUB AND HEROKU ####
git_dir = '/Users/mdonnelly/repos/COVIDoutlook/'
git_commit_cmd = 'git commit -am "Auto update on {}"'.format(
    pd.Timestamp.today().strftime("%Y-%m-%d at %I:%M %p"))
print(git_commit_cmd)
status_out = subprocess.check_output('git status', cwd=git_dir, shell=True).decode()
print(status_out)
status_out = subprocess.check_output('git add download/*.csv', cwd=git_dir, shell=True).decode()
commit_out = subprocess.check_output(git_commit_cmd, cwd=git_dir, shell=True).decode()
print(commit_out)
push_out = subprocess.check_output('git push', cwd=git_dir, shell=True).decode()
# push_out = subprocess.check_output('git push heroku master', cwd=git_dir, shell=True).decode()
print(push_out)

os.system('say -v "Victoria" "COVID Outlook dot info has been updated."')
#####################################