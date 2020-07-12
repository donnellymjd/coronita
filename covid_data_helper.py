import pandas as pd
import numpy as np

# From Roger Allen https://gist.github.com/rogerallen/1583593
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
idx = pd.IndexSlice

def get_nys_region(): 
    gsheet_nys = 'https://docs.google.com/spreadsheets/d/1yidLf5CUEsdFpaYSF5is_KSJ5M5Okm4p3c7eduBkM8s/export?format=csv&gid=1928535373'
    df_nys_region_raw = pd.read_csv(gsheet_nys, skiprows=2)

    df_nys_region = df_nys_region_raw.copy()
    df_nys_region['dt'] = pd.to_datetime(df_nys_region.Date)
    df_nys_region = df_nys_region.set_index('dt').iloc[:,2:]
    df_nys_region = df_nys_region.stack().reset_index()

    df_nys_region[['ny_region','metric']] = df_nys_region['level_1'].str.split(':', expand=True)

    df_nys_region = df_nys_region.rename(columns={0:'value'})[['dt','ny_region','metric','value']]
    df_nys_region = df_nys_region.set_index(['dt','ny_region','metric'])
    df_nys_region = df_nys_region.unstack(2)['value']
    df_nys_region = df_nys_region.swaplevel().sort_index()
    df_nys_region.columns = df_nys_region.columns.str.lstrip()
    df_nys_region = df_nys_region.apply(lambda x: 
                        x.str.replace(',','').str.replace('%','').replace('#DIV/0!',np.nan).astype(float), axis=1)
    return df_nys_region

def get_nyt_counties():
    raw_reporting = pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv')
    df_reporting = raw_reporting
    df_reporting['fips']= df_reporting['fips'].astype(str).replace('\.0', '', regex=True).str.zfill(5)
    df_reporting['dt'] = pd.to_datetime(df_reporting.date)
    df_reporting = df_reporting.drop(columns=['date'])
    df_reporting = df_reporting.set_index(['dt','state','county']).sort_index()
    return df_reporting


def get_jhu_counties():
    df_jhu_counties_cases_raw = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

    df_jhu_counties_deaths_raw = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

    df_jhu_counties = pd.concat(
        [process_jhu_counties(df_jhu_counties_cases_raw, 'cases'),
         process_jhu_counties(df_jhu_counties_deaths_raw, 'deaths')], axis=1)

    return df_jhu_counties

def process_jhu_counties(df_jhu_counties, series_name):
    df_jhu_counties['FIPS'] = df_jhu_counties['FIPS'].dropna().astype(str).replace('\.0', '', regex=True).str.zfill(5)
    dropcols = [x for x in df_jhu_counties.columns if x in
                ['UID','iso2','iso3','code3','Country_Region','Lat','Long_','Combined_Key','Population']]

    df_jhu_counties = df_jhu_counties.drop(dropcols, axis=1)

    df_jhu_counties = df_jhu_counties.rename(columns={'FIPS':'fips','Admin2':'county','Province_State':'state'})
    df_jhu_counties = df_jhu_counties.set_index(['state','county','fips'])
    df_jhu_counties = df_jhu_counties.stack().reset_index().rename(columns={'level_3':'dt',0:series_name})
    df_jhu_counties = df_jhu_counties.set_index(['dt','state','county','fips']).sort_index()
    return df_jhu_counties

def get_nyregionmap():
    df_reporting = get_nyt_counties()
    
    ny_all_counties = df_reporting.reset_index()[df_reporting.reset_index().state=='New York'
                                            ]['county'].unique()
    rest_of_nys = [cty for cty in ny_all_counties if cty not in ['Nassau', 'Suffolk','New York City',
                                                                'Westchester','Rockland']]
    nys_regions_map = {'Long Island':['Nassau', 'Suffolk'],
     'NYC':['New York City'],
     'Westchester & Rockland':['Westchester','Rockland'],
     'Rest of NYS':rest_of_nys}
    return nys_regions_map

def reporting_by_nys_region():
    df_reporting = get_nyt_counties()
    nys_regions_map = get_nyregionmap()
    
    df_reporting_fmt = pd.DataFrame()
    
    for col in ['cases','deaths']:
        df_tmp = pd.DataFrame()
        for reg_str, county_list in nys_regions_map.items():
            df_tmp[reg_str] = df_reporting[col].unstack([1,2])['New York'][county_list].sum(axis=1).fillna(0)
        df_tmp['NYS'] = df_reporting[col].unstack([1,2])['New York'].sum(axis=1).fillna(0)
        df_reporting_fmt[col] = df_tmp.stack()
    df_reporting_fmt = df_reporting_fmt.unstack()
    
    return df_reporting_fmt

def get_nycdoh_data():
    df_nycdoh_raw = pd.read_csv('https://github.com/nychealth/coronavirus-data/raw/master/case-hosp-death.csv')
    df_nycdoh = df_nycdoh_raw
    df_nycdoh['dt'] = pd.to_datetime(df_nycdoh['DATE_OF_INTEREST'])
    df_nycdoh = df_nycdoh.drop(columns=['DATE_OF_INTEREST'])
    df_nycdoh = df_nycdoh.set_index('dt').sort_index()
    return df_nycdoh

def get_nycdoh_boro():
    df_nycdoh_raw = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/boro/boroughs-case-hosp-death.csv')
    df_nycdoh = df_nycdoh_raw
    df_nycdoh['dt'] = pd.to_datetime(df_nycdoh['DATE_OF_INTEREST'])
    df_nycdoh = df_nycdoh.drop(columns=['DATE_OF_INTEREST'])
    df_nycdoh = df_nycdoh.set_index('dt').sort_index()
    df_nycdoh = df_nycdoh.stack().reset_index().rename(columns={'level_1': 'metric', 0: 'value'})
    df_nycdoh['county'] = df_nycdoh.metric.apply(lambda x: x[:2]).replace(
        {'BK': 'Kings', 'QN': 'Queens', 'SI': 'Richmond', 'MN': 'Manhattan', 'BX': 'Bronx'})
    df_nycdoh['metric'] = df_nycdoh.metric.apply(lambda x: x[3:])
    df_nycdoh = df_nycdoh.set_index(['dt', 'county', 'metric']).unstack('metric')['value']
    df_nycdoh = df_nycdoh.rename(columns={'CASE_COUNT':'cases_daily',
                                          'DEATH_COUNT':'deaths_daily',
                                          'HOSPITALIZED_COUNT':'hosp_admits'})

    # df_nycdoh['state'] = 'NY'
    # df_census = get_census_pop()
    # df_nycdoh = pd.merge(df_nycdoh, df_census.loc[df_census['SUMLEV'] == 50,
    #                                               ['state', 'county', 'fips']], how='inner', on=['state', 'county'])
    # df_nycdoh['state'] = 'New York'
    return df_nycdoh

def get_nysdoh_data():
    # df_nys_pub = pd.read_json('https://health.data.ny.gov/resource/xdss-u53e.json')

    df_nys_pub = pd.read_csv('https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD')
    df_nys_pub.columns = [x.lower().replace(' ', '_') for x in df_nys_pub.columns]

    df_nys_pub['dt'] = pd.to_datetime(df_nys_pub['test_date'])
    df_nys_pub = df_nys_pub.set_index(['county', 'dt']).drop(columns='test_date').sort_index()
    return df_nys_pub

def get_complete_county_data():
    df_nys_pub = get_nysdoh_data()
    df_nys_pub = df_nys_pub.reset_index()
    df_nys_pub['state'] = 'NY'

    df_census = get_census_pop()

    df_nys_pub = pd.merge(df_nys_pub, df_census.loc[df_census['SUMLEV'] == 50,
                                                    ['state', 'county', 'fips']], how='inner', on=['state', 'county'])
    df_nys_pub['state'] = 'New York'
    df_nys_pub = df_nys_pub.rename(columns={'cumulative_number_of_positives': 'cases'})
    df_nys_pub = df_nys_pub[['dt', 'state', 'county', 'fips', 'cases']]

    df_counties = get_nyt_counties()
    df_counties = df_counties.reset_index()
    df_counties = df_counties[~(df_counties.county == 'New York City')]

    nycounties_notin_nyt = [x for x in df_nys_pub.fips.unique() if x not in df_counties.fips.unique()]

    df_nycdoh = get_nycdoh_boro()
    df_nycdoh = df_nycdoh.unstack('county').cumsum().stack('county').rename(columns={'deaths_daily': 'deaths'})
    df_nycdoh = df_nycdoh[['deaths']].reset_index()

    df_notin_nyt = pd.merge(df_nys_pub[df_nys_pub.fips.isin(nycounties_notin_nyt)],
                            df_nycdoh, how='outer', on=['dt', 'county'])

    df_counties = pd.concat([df_counties.reset_index(), df_notin_nyt], axis=0)

    df_counties = pd.merge(
        df_census.loc[df_census['SUMLEV'] == 50, ['state','county','fips','pop2019']],
        df_counties.reset_index()[['dt', 'fips', 'cases', 'deaths']],
        on='fips', how='inner')
    df_counties['cases_per100k'] = df_counties['cases'].mul(1e5).div(df_counties['pop2019'])

    # df_counties = df_counties.set_index(['dt','state','county','fips']).sort_index()

    df_goog_mob_cty = get_goog_mvmt_cty()
    df_counties = pd.merge(df_counties[[x for x in df_counties.columns if x not in ['state','county']]],
                           df_goog_mob_cty, on=['dt','fips'], how='outer')
    df_counties = pd.merge(
        df_census.loc[df_census['SUMLEV'] == 50, ['state', 'county', 'fips']],
        df_counties,
        on='fips', how='inner')
    df_counties = df_counties.set_index(['dt', 'state', 'county', 'fips']).sort_index()

    return df_counties

def get_covid19_tracking_data():
    df_st_testing_raw = pd.read_csv(
    'https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv')
    df_st_testing = df_st_testing_raw
    df_st_testing['dt'] = pd.to_datetime(df_st_testing['date'], format="%Y%m%d")
    print("State Testing Data Last Observation: ", df_st_testing.date.max())
    df_st_testing = df_st_testing.rename(columns={'state':'code'})
    df_st_testing = df_st_testing.set_index(['code','dt']).sort_index()
    return df_st_testing

def get_census_pop():
    df_census_raw = pd.read_csv(
    'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv', 
    encoding = "ISO-8859-1")
    df_census = df_census_raw.copy()
    df_census['county'] = df_census.CTYNAME.str.replace(' County','').str.replace(' Parish','')
    df_census['fips'] = df_census.STATE.apply('{:0>2}'.format).astype(str).str.cat(
        df_census.COUNTY.apply('{:0>3}'.format).astype(str))
    df_census = df_census.rename(columns={'STNAME':'state','POPESTIMATE2019':'pop2019'})
    df_census['pop2019'] = pd.to_numeric(df_census.pop2019)
    df_census = df_census[['state','county','fips','SUMLEV', 'REGION','DIVISION', 'pop2019']]
    df_census['state'] = df_census['state'].replace(us_state_abbrev)
    return df_census

def get_goog_mvmt_us():
    df_goog_mob_raw = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',
                                  low_memory=False)
    df_goog_mob_us = df_goog_mob_raw[df_goog_mob_raw.country_region_code == 'US'].copy()
    df_goog_mob_us = df_goog_mob_us.rename(columns={'sub_region_1': 'state',
                                                    'sub_region_2': 'county',
                                                    'date': 'dt',
                                                    'census_fips_code':'fips'})
    df_goog_mob_us['dt'] = pd.to_datetime(df_goog_mob_us['dt'])
    df_goog_mob_us['state'] = df_goog_mob_us['state'].replace(us_state_abbrev)
    df_goog_mob_us['county'] = df_goog_mob_us['county'].str.replace(' Parish', '', regex=True
                                                                    ).replace(' County', '', regex=True)
    df_goog_mob_us['fips'] = df_goog_mob_us['fips'].fillna(0).astype(int).apply('{:0>5}'.format)
    return df_goog_mob_us

def get_goog_mvmt_cty():
    df_goog_mob_us = get_goog_mvmt_us()
    # df_census = get_census_pop()
    # df_census = df_census.loc[df_census['SUMLEV'] == 50, ['state','county','fips']]
    mobility_cols = ['retail_and_recreation_percent_change_from_baseline',
                     'grocery_and_pharmacy_percent_change_from_baseline',
                     'parks_percent_change_from_baseline',
                     'transit_stations_percent_change_from_baseline',
                     'workplaces_percent_change_from_baseline',
                     'residential_percent_change_from_baseline']

    key_cols = ['dt','fips']

    # df_goog_mob_cty = df_goog_mob_us[~(df_goog_mob_us['state'].isnull()) & ~(df_goog_mob_us['county'].isnull())]
    df_goog_mob_cty = df_goog_mob_us[df_goog_mob_us['fips'] != '00000']
    df_goog_mob_cty = df_goog_mob_cty[key_cols + mobility_cols]
    # df_goog_mob_cty = pd.merge(df_census, df_goog_mob_cty, on=['state', 'county'])
    # df_goog_mob_cty = df_goog_mob_cty.set_index(key_cols).sort_index()
    return df_goog_mob_cty

def get_goog_mvmt_state():
    df_goog_mob_us = get_goog_mvmt_us()
    # df_census = get_census_pop()[['state', 'SUMLEV', 'REGION', 'DIVISION', 'pop2019']]
    mobility_cols = ['retail_and_recreation_percent_change_from_baseline',
                     'grocery_and_pharmacy_percent_change_from_baseline',
                     'parks_percent_change_from_baseline',
                     'transit_stations_percent_change_from_baseline',
                     'workplaces_percent_change_from_baseline',
                     'residential_percent_change_from_baseline']

    key_cols = ['state', 'dt']

    df_goog_mob_state = df_goog_mob_us[~(df_goog_mob_us.state.isnull()) & (df_goog_mob_us.county.isnull())].copy()
    df_goog_mob_state = df_goog_mob_state[key_cols + mobility_cols]
    # df_goog_mob_state = pd.merge(df_census[df_census.SUMLEV == 40], df_goog_mob_state, on=['state'])
    # df_goog_mob_state = df_goog_mob_state.set_index(df_census.columns.to_list() + ['dt']).sort_index()
    df_goog_mob_state = df_goog_mob_state.set_index(key_cols)
    return df_goog_mob_state

def get_state_policy_events():
    import requests
    import re
    from bs4 import BeautifulSoup

    url = 'https://www.kff.org/report-section/state-data-and-policy-actions-to-address-coronavirus-sources/'
    res = requests.get(url)
    html_page = res.content

    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script']

    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)

    rawlist = list(map(str.strip, output.split('\n')))
    outlist = []
    this_state = ''
    # us_states = [x.upper() for x in us_state_abbrev.keys()]
    us_states = {k.upper(): v for k, v in us_state_abbrev.items()}

    for linenum in range(len(rawlist)):
        thisstr = rawlist[linenum]

        if thisstr.strip().upper() in us_states.keys():
            change_dir = 'restricting'
            this_state_abbrev = us_states[thisstr.strip().upper()]
            this_state = abbrev_us_state[this_state_abbrev]
        elif thisstr.strip()[:6] == 'Easing':
            change_dir = 'easing'
        elif this_state != '' and thisstr.strip()[:4] not in ['http', '']:

            #         idx_before_urls = thisstr.find(':')
            idx_before_urls = re.search("[A-Za-z]", thisstr).start()

            dates = re.findall(r'\d+/\d+', thisstr[:idx_before_urls])
            if len(dates) > 0:
                last_dt = dates[-1]

                ld_idx = thisstr.find(last_dt)

                name_url = thisstr[ld_idx + len(last_dt):]
                l_name_url = re.split(":", name_url, 1)

                name = l_name_url[0].strip()
                if len(l_name_url) > 1:
                    urls = l_name_url[1].strip()

                outlist.append([this_state, this_state_abbrev, dates[0] + '/2020', dates, name, change_dir, urls])

    df_out = pd.DataFrame(outlist,
                          columns=['state', 'state_code', 'dt', 'all_dates',
                                   'event_name', 'social_distancing_direction', 'urls'])
    df_out.loc[df_out['social_distancing_direction'] == 'easing', 'event_name'] = 'Easing: ' + df_out['event_name']

    df_holidays = pd.read_csv(
        'https://gist.githubusercontent.com/shivaas/4758439/raw/b0d3ddec380af69930d0d67a9e0519c047047ff8/US%2520Bank%2520holidays',
        header=None, names=['idx', 'dt', 'event_name'], usecols=[1, 2])

    df_holidays['state'] = 'US'
    df_holidays['state_code'] = 'US'
    df_holidays['social_distancing_direction'] = 'holiday'

    df_out = pd.concat([df_out, df_holidays])

    df_out['dt'] = pd.to_datetime(df_out['dt'])

    return df_out

def get_counties_geo():
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    return counties