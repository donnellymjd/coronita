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

### Settings and Functions for Personal Website ###
plt.style.use('ggplot')

def footnote_str_maker():
    footnote_str = 'www.COVIDoutlook.info | twtr: @COVIDoutlook\nChart created on {}'.format(
        pd.Timestamp.today().strftime("%d %b, %Y at %I:%M %p"))
    return footnote_str


def add_plotly_footnote(fig):
    fig.update_layout(
                  annotations=[
                      dict(x = 0, y = -0.06, showarrow = False,
                           xref='paper', yref='paper',
                           xanchor='left', yanchor='auto', xshift=0, yshift=0,
                            text='<a href="http://{0}">{0}</a> | twtr: <a href="https://twitter.com/COVIDoutlook">@COVIDoutlook</a> | '.format(
                           'www.COVIDoutlook.info')
                          ),
                      dict(x = 0, y = -0.09, showarrow = False,
                           xref='paper', yref='paper',
                           xanchor='left', yanchor='auto', xshift=0, yshift=0,
                           text='Chart created on {}'.format(pd.Timestamp.today().strftime("%d %b %Y"))
                          )
                  ]
                 )
    return fig

state_md_template = '''---
title: {0}
layout: noheader
statecode: {1}
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
df_goog_mob_us = df_goog_mob_us[df_goog_mob_us.state.isnull()].set_index('dt')

df_goog_mob_state = get_goog_mvmt_state()

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

fig = ch_exposure_prob(df_fore_allstates,
                       df_census[df_census.SUMLEV == 40].set_index('state')['pop2019'])
fig = add_plotly_footnote(fig)
fig.write_html('../COVIDoutlook/forecasts/plotly/ch_exposure_prob.html', include_plotlyjs='cdn')

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
image: duotone2.png
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
scenario_name = 'No Change in Future Rᵗ Until 20% Hospital Capacity Trigger'
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

p_pop_share = bk_population_share(df_agg, model_dict, param_str, chart_title)
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
image: duotone-us.png
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
{% for chart in div[2:-1] %}
    <div class="w3-container">
        {{ chart }}
    </div>
{% endfor %}
{{ exposure_prob }}
{{ case_change }}
<div class="w3-container">
    {{ div[-1] }}
</div>
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
image: duotone4.png
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
           'ch_exposed_infectious', 'ch_hosp',
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
 'ch_hosp': ch_hosp,
 'ch_population_share': ch_population_share,
 'ch_cumul_infections': ch_cumul_infections,
 'ch_daily_exposures': ch_daily_exposures,
 'ch_hosp_admits': ch_hosp_admits,
 'ch_daily_deaths': ch_daily_deaths}

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
    fig.write_html('../COVIDoutlook/forecasts/plotly/{}_casepercap_cnty_map.html'.format(
        model_dict['region_code']), include_plotlyjs='cdn')

    try:
        pio.orca.shutdown_server()
        fig.write_image(cover_file, scale=2)
    except:
        pio.orca.shutdown_server()

    for ch_name, ch_fn in d_chart_fns.items():
        try:
            ax = ch_fn(model_dict)
            filename = '../COVIDoutlook/assets/images/covid19/{}_{}.png'.format(
                model_dict['region_code'], ch_name)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            os.system('optipng {} &'.format(filename))
        except:
            print('Couldn\'t create {} {} chart.'.format(model_dict['region_code'], ch_name))


    l_content = ['### How Fast is COVID-19 Currently Spreading?']

    for thischart in l_charts:
        if thischart == 'ch_statemap':
            l_content.append('{{% include_relative plotly/{}_casepercap_cnty_map.html %}}'.format(state_code))
        else:
            l_content.append("<img src='/assets/images/covid19/{}_{}.png'>".format(
                state_code, thischart))

        if thischart in dict_ch_defs.keys():
            l_content.append(dict_ch_defs[thischart]+'\n- - - -')

    l_content.insert(15, '### Model and Forecast Results')


    final_md = state_md_template.format(model_dict['region_name'], state_code, '\n'.join(l_content))

    filename = "../COVIDoutlook/forecasts/{}.md".format(state_code)

    with open(filename, "w") as file:
        file.write(final_md)
#####################################


#### POST FORECAST DATA TO COVIDOUTLOOK ####

download_header = """---
layout: page
title: Data
image: duotone5.png
---
This data page, like the rest of the site, is brand new. Unfortunately, it's not fully functional yet. Please check back here soon for downloadable datasets of the forecasts and other data displayed on this website.

### Forecast Downloads (comma separated values)
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
    output_md.append(' - [Forecast published on {0}]({1})'.format(this_file[1], this_file[0][15:]))

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
push_out = subprocess.check_output('git push heroku master', cwd=git_dir, shell=True).decode()
print(push_out)

os.system('say -v "Victoria" "COVID Outlook dot info has been updated."')
#####################################