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
    df_nycdoh = df_nycdoh.set_index('dt')
    return df_nycdoh

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
