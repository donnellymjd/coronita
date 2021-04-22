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
    print('Got NYT county level data.')
    return df_reporting


def get_jhu_counties():
    df_jhu_counties_cases_raw = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

    df_jhu_counties_deaths_raw = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

    df_jhu_counties = pd.concat(
        [process_jhu_counties(df_jhu_counties_cases_raw, 'cases'),
         process_jhu_counties(df_jhu_counties_deaths_raw, 'deaths')], axis=1)
    print('Got JHU county level data.')
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
    # df_nycdoh_raw = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/boro/boroughs-case-hosp-death.csv')
    df_nycdoh_raw = pd.read_csv(
        'https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
    df_nycdoh = df_nycdoh_raw
    df_nycdoh['dt'] = pd.to_datetime(df_nycdoh['date_of_interest'])
    df_nycdoh = df_nycdoh.drop(columns=['date_of_interest'])
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
    print('Got NYC DOH data')
    return df_nycdoh

def get_nysdoh_data():
    # df_nys_pub = pd.read_json('https://health.data.ny.gov/resource/xdss-u53e.json')

    df_nys_pub = pd.read_csv('https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD')
    df_nys_pub.columns = [x.lower().replace(' ', '_') for x in df_nys_pub.columns]

    df_nys_pub['dt'] = pd.to_datetime(df_nys_pub['test_date'])
    df_nys_pub = df_nys_pub.set_index(['county', 'dt']).drop(columns='test_date').sort_index()
    print('Got NYS DOH data')
    return df_nys_pub

def get_complete_county_data(df_census, df_goog_mob_us):
    df_nys_pub = get_nysdoh_data()
    df_nys_pub = df_nys_pub.reset_index()
    df_nys_pub['state'] = 'NY'

    # df_census = get_census_pop()

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

    # df_goog_mob_us = get_goog_mvmt_us()
    df_goog_mob_cty = get_goog_mvmt_cty(df_goog_mob_us)
    df_counties = pd.merge(df_counties[[x for x in df_counties.columns if x not in ['state','county']]],
                           df_goog_mob_cty, on=['dt','fips'], how='outer')
    df_counties = pd.merge(
        df_census.loc[df_census['SUMLEV'] == 50, ['state', 'county', 'fips']],
        df_counties,
        on='fips', how='inner')

    df_counties = df_counties.set_index(['dt', 'state', 'county', 'fips']).sort_index()
    print('Got Complete County Time Series Data')
    return df_counties

def get_covid19_tracking_data_old():
    df_st_testing_raw = pd.read_csv(
    # 'https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv')
        'https://api.covidtracking.com/v1/states/daily.csv')
    df_st_testing = df_st_testing_raw
    df_st_testing['dt'] = pd.to_datetime(df_st_testing['date'], format="%Y%m%d")
    print("State Testing Data Last Observation: ", df_st_testing.date.max())
    df_st_testing = df_st_testing.rename(columns={'state':'code'})
    df_st_testing = df_st_testing.set_index(['code','dt']).sort_index()
    print('Got COVID19 Tracking Data')
    return df_st_testing

def get_covid19_tracking_data():
    df_st_testing_raw = pd.read_csv(
        # 'https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv')
        'https://api.covidtracking.com/v1/states/daily.csv')
    df_st_testing = df_st_testing_raw
    df_st_testing['dt'] = pd.to_datetime(df_st_testing['date'], format="%Y%m%d")
    print("State Testing Data Last Observation: ", df_st_testing.date.max())
    df_st_testing = df_st_testing.rename(columns={'state': 'code'})
    df_st_testing = df_st_testing.set_index(['code', 'dt']).sort_index()
    print('Got COVID19 Tracking Data')

    df_cdc_raw = pd.read_csv('https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD')
    rename_cols = {'state': 'code', 'submission_date': 'dt', 'tot_death': 'death', 'tot_cases': 'cases'}
    df_cdc = df_cdc_raw[rename_cols.keys()]
    df_cdc = df_cdc.rename(columns=rename_cols)
    df_cdc['dt'] = pd.to_datetime(df_cdc['dt']).dt.normalize()
    df_cdc['code'] = df_cdc['code'].replace('NYC', 'NY')
    df_cdc = df_cdc.groupby(['code', 'dt']).sum().sort_index()
    print('Got CDC Death Data')

    # https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb
    df_hhs_tests_raw = pd.read_csv(
        'https://healthdata.gov/api/views/j8mb-icvb/rows.csv?accessType=DOWNLOAD')
    df_hhs_tests = df_hhs_tests_raw.copy()

    df_hhs_tests = df_hhs_tests.rename(columns={'state': 'code', 'date': 'dt'})
    df_hhs_tests['dt'] = pd.to_datetime(df_hhs_tests['dt']).dt.normalize()
    df_hhs_tests = df_hhs_tests.set_index(['code', 'dt', 'overall_outcome'])['total_results_reported']
    df_hhs_tests = df_hhs_tests.unstack('overall_outcome')
    df_hhs_tests['posNeg'] = df_hhs_tests[['Inconclusive', 'Negative', 'Positive']].sum(axis=1)
    df_hhs_tests = df_hhs_tests.rename(columns={'Positive': 'positive'})
    df_hhs_tests = df_hhs_tests[['positive', 'posNeg']]
    print('Got HHS testing data')

    df_hhs_hosp_raw = pd.read_csv('https://beta.healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD')
    df_hhs_hosp = df_hhs_hosp_raw
    df_hhs_hosp = df_hhs_hosp.rename(columns={'state': 'code', 'date': 'dt'})
    df_hhs_hosp['dt'] = pd.to_datetime(df_hhs_hosp['dt']).dt.normalize()
    df_hhs_hosp = df_hhs_hosp.set_index(['code', 'dt'])
    df_hhs_hosp['hospitalizedIncrease'] = df_hhs_hosp[[
        'previous_day_admission_adult_covid_confirmed',
        'previous_day_admission_adult_covid_suspected',
        'previous_day_admission_pediatric_covid_confirmed',
        'previous_day_admission_pediatric_covid_suspected']].sum(axis=1)
    df_hhs_hosp['hospitalizedCurrently'] = df_hhs_hosp[[
        'total_adult_patients_hospitalized_confirmed_and_suspected_covid',
        'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid']].sum(axis=1)
    df_hhs_hosp = df_hhs_hosp[['hospitalizedIncrease', 'hospitalizedCurrently']].sort_index()
    print('Got HHS hospital data.')

    df_hosp_joint = pd.concat([
        df_st_testing[['hospitalizedIncrease', 'hospitalizedCurrently']].unstack(['code']).loc[:'2020-08-31'].stack(
            'code'),
        df_hhs_hosp[['hospitalizedIncrease', 'hospitalizedCurrently']].unstack(['code']).loc['2020-09-01':].stack(
            'code')]
    )
    df_hosp_joint = df_hosp_joint.swaplevel().sort_index()

    df_st_testing_new = pd.merge(df_cdc['death'], df_hhs_tests, how='outer', left_index=True, right_index=True)
    df_st_testing_new = pd.merge(df_st_testing_new, df_hosp_joint, how='outer', left_index=True, right_index=True)
    return df_st_testing_new

def get_census_pop():
    df_census_raw = pd.read_csv(
        'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv',
        encoding = "ISO-8859-1")
    df_adults = pd.read_csv(
        'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/state/detail/SCPRC-EST2019-18+POP-RES.csv')
    df_census = df_census_raw.copy()
    df_census['county'] = df_census.CTYNAME.str.replace(' County', '').str.replace(' Parish', '')
    df_census['fips'] = df_census.STATE.apply('{:0>2}'.format).astype(str).str.cat(
        df_census.COUNTY.apply('{:0>3}'.format).astype(str))
    df_census = pd.merge(df_census, df_adults.loc[df_adults.SUMLEV == 40, ['STATE', 'PCNT_POPEST18PLUS']], on='STATE')
    df_census = df_census.rename(columns={'STNAME': 'state', 'POPESTIMATE2019': 'pop2019'})
    df_census['pop2019'] = pd.to_numeric(df_census.pop2019)
    df_census['pop2019_18plus'] = df_census['pop2019'] * df_census['PCNT_POPEST18PLUS'] / 100
    df_census = df_census[['state', 'county', 'fips', 'SUMLEV', 'REGION', 'DIVISION', 'pop2019', 'pop2019_18plus']]
    df_census['state'] = df_census['state'].replace(us_state_abbrev)
    print('Got Census Data')
    return df_census

def get_goog_mvmt_us():
    # df_goog_mob_raw = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',
    #                               low_memory=False)
    from io import BytesIO
    from zipfile import ZipFile
    import requests

    url = requests.get('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')
    zipfile = ZipFile(BytesIO(url.content))
    with zipfile.open('2020_US_Region_Mobility_Report.csv') as f:
        df_goog_mob20_raw = pd.read_csv(f, low_memory=False)

    with zipfile.open('2021_US_Region_Mobility_Report.csv') as f:
        df_goog_mob21_raw = pd.read_csv(f, low_memory=False)

    df_goog_mob_raw = pd.concat([df_goog_mob20_raw, df_goog_mob21_raw])

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
    print('Got Google Movement Data')
    return df_goog_mob_us

def get_goog_mvmt_cty(df_goog_mob_us):
    # df_goog_mob_us = get_goog_mvmt_us()
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

def get_goog_mvmt_state(df_goog_mob_us):
    # df_goog_mob_us = get_goog_mvmt_us()
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
            if re.search("[A-Za-z]", thisstr) != None:
                idx_before_urls = re.search("[A-Za-z]", thisstr).start()
            else:
                idx_before_urls = None

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
    print('Got KFF Policy dates')
    return df_out

def get_counties_geo():
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    print('Got counties geo json')
    return counties

def get_hhs_hosp():
    df_hhs_hosp = pd.read_csv('https://healthdata.gov/api/views/jjp9-htie/rows.csv?accessType=DOWNLOAD')
    df_hhs_hosp['state'] = df_hhs_hosp['state'].replace('CW', 'US')
    df_hhs_hosp['dt'] = pd.to_datetime(df_hhs_hosp['collection_date']).dt.normalize()
    df_hhs_hosp = df_hhs_hosp.set_index(['state', 'dt']).sort_index()

    # df_hhs_hosp['Total Inpatient Beds'] = df_hhs_hosp['Total Inpatient Beds'].str.replace(',', '')
    df_hhs_hosp['Total Inpatient Beds'] = pd.to_numeric(df_hhs_hosp['Total Inpatient Beds'], errors='coerce')

    # df_hhs_hosp['Inpatient Beds Occupied Estimated'] = df_hhs_hosp['Inpatient Beds Occupied Estimated'].str.replace(',', '')
    df_hhs_hosp['Inpatient Beds Occupied Estimated'] = pd.to_numeric(df_hhs_hosp['Inpatient Beds Occupied Estimated'],
                                                                     errors='coerce')

    df_hhs_hosp['hosp_beds_avail'] = df_hhs_hosp['Total Inpatient Beds'] - df_hhs_hosp[
        'Inpatient Beds Occupied Estimated']
    print('Got HHS hospitalization data.')
    return df_hhs_hosp

def get_can_data():
    from my_can_apikey import can_apikey
    df_can_raw = pd.read_csv(f'https://api.covidactnow.org/v2/states.timeseries.csv?apiKey={can_apikey}')
    df_can = df_can_raw.copy()

    df_can['dt'] = pd.to_datetime(df_can['date'])
    df_can = df_can.set_index(['state', 'dt'])
    print('Got COVID Act Now data.')
    return df_can


def get_can_counties_data():
    from my_can_apikey import can_apikey
    df_cancounties = pd.read_csv(f'https://api.covidactnow.org/v2/counties.timeseries.csv?apiKey={can_apikey}')
    df_cancounties['dt'] = pd.to_datetime(df_cancounties['date'])
    df_cancounties['fips'] = df_cancounties['fips'].astype(str).replace('\.0', '', regex=True).str.zfill(5)

    df_cancounties = df_cancounties.set_index(['state', 'fips', 'dt'])

    df_cancounties = df_cancounties[['actuals.cases',
                                     'actuals.deaths', 'actuals.positiveTests', 'actuals.negativeTests',
                                     'actuals.contactTracers', 'actuals.hospitalBeds.capacity',
                                     'actuals.hospitalBeds.currentUsageTotal',
                                     'actuals.hospitalBeds.currentUsageCovid',
                                     'actuals.hospitalBeds.typicalUsageRate', 'actuals.icuBeds.capacity',
                                     'actuals.icuBeds.currentUsageTotal',
                                     'actuals.icuBeds.currentUsageCovid', 'actuals.icuBeds.typicalUsageRate',
                                     'actuals.newCases', 'actuals.vaccinesDistributed',
                                     'actuals.vaccinationsInitiated', 'actuals.vaccinationsCompleted',
                                     'metrics.testPositivityRatio', 'metrics.testPositivityRatioDetails',
                                     'metrics.caseDensity', 'metrics.contactTracerCapacityRatio',
                                     'metrics.infectionRate', 'metrics.infectionRateCI90',
                                     'metrics.icuHeadroomRatio', 'metrics.icuHeadroomDetails',
                                     'metrics.icuCapacityRatio', 'riskLevels.overall',
                                     'metrics.vaccinationsInitiatedRatio',
                                     'metrics.vaccinationsCompletedRatio', 'actuals.newDeaths',
                                     'actuals.vaccinesAdministered']]
    return df_cancounties

def get_vax_hesitancy_data():
    df_vax_hes = pd.read_csv('./data/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates.csv')
    newcols = {
        'FIPS Code': 'fips',
        'County Name': 'county',
        'State Code': 'state',
        'Estimated hesitant': 'est_vax_hes',
        'Estimated strongly hesitant': 'est_vax_hes_strong'
    }
    df_vax_hes = df_vax_hes[newcols.keys()].rename(columns=newcols)
    df_vax_hes['fips'] = df_vax_hes['fips'].astype(str).replace('\.0', '', regex=True).str.zfill(5)
    df_census = get_census_pop()
    df_vax_hes = pd.merge(df_vax_hes, df_census[['fips', 'pop2019', 'pop2019_18plus']], on='fips')
    df_vax_hes['est_vax_hes_pop_18plus'] = df_vax_hes['est_vax_hes'] * df_vax_hes['pop2019_18plus']
    df_vax_hes['est_vax_hes_strong_pop_18plus'] = df_vax_hes['est_vax_hes_strong'] * df_vax_hes['pop2019_18plus']

    print('Got HHS Vaccine Hesitancy Data.')

    return df_vax_hes
