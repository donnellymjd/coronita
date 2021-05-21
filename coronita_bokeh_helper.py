import pandas as pd
import numpy as np
from collections import OrderedDict
import io

from bokeh.plotting import figure, show
from bokeh.io import reset_output, output_notebook, curdoc, output_file, save
from bokeh.themes import built_in_themes
from bokeh.layouts import row, column, grid
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, Label, LinearAxis, Range1d, \
    Span, DatetimeTickFormatter, CustomJS, Select, Button, Patch, Legend, Div, Title, FactorRange
from bokeh.embed import components, autoload_static
from bokeh.resources import INLINE, CDN

from jinja2 import Template

bk_theme = 'light_minimal'

def add_bokeh_footnote(p):
    msg1 = 'www.COVIDoutlook.info | twtr: @COVIDoutlook'
    msg2 = 'Chart created on {}'.format(pd.Timestamp.today().strftime("%b %d '%y"))

    label_opts = dict(
        x=0, y=0,
        x_units='screen', y_units='screen',
        text_font_size='80%',
        text_color='black',
        render_mode='canvas'
    )

    caption1 = Label(text=msg1, **label_opts)
    caption2 = Label(text=msg2, **label_opts)
    p.add_layout(caption1, 'below')
    p.add_layout(caption2, 'below')
    return p

def bk_title(p, title="", subtitle=""):
    p.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="100%"), 'above')
    p.add_layout(Title(text=title, text_font_style="bold", text_font_size="125%"), 'above')
    return p

def bk_legend(p, location='center', orientation='horizontal', font_size=1.35):
    p.legend.title = 'Interactive Legend'
    p.legend.title_text_font_style = "bold"
    p.legend.title_text_font_size = '100%' #str(font_size*1.1)+'em'
    p.legend.title_text_color = "black"
    p.legend.label_text_font_size = '75%' #str(font_size)+'em'
    p.legend.glyph_height = 15
    p.legend.label_height = 8
    p.legend.glyph_width = 8
    p.legend.spacing = 5
    p.legend.padding = 5
    p.legend.background_fill_alpha = 0.9
    p.legend.click_policy = "hide"
    p.legend.location = location
    p.legend.orientation = orientation

    return p

def bk_overview_layout(p, num_in_row=1, min_height=360):
    p = bk_legend(p, location='center', orientation='horizontal')
    # p.legend.visible = False
    p.add_layout(p.legend[0], 'below')

    p.title.text_font_size = '100%'
    p.title.align = 'left'

    p.xaxis.formatter=DatetimeTickFormatter(days=["%b %d"], months=["%b '%y"])
    p.yaxis.axis_label_text_font_size = '100%'
    p.yaxis.major_label_text_font_size = '80%'
    p.yaxis.major_tick_line_alpha = .9
    p.yaxis.minor_tick_line_alpha = 0

    p.toolbar_location = 'right'
    p.toolbar.autohide = True

    p = add_bokeh_footnote(p)

    if num_in_row > 1:
        p.plot_height = int(min_height*1.2)
        p.plot_width = p.plot_height
        p.sizing_mode = 'scale_both'
    else:
        p.plot_height = int(min_height*1.2)
        p.plot_width = int(min_height*5/3)
        p.sizing_mode = 'scale_width'

    #Possible values are "fixed", "scale_width", "scale_height", "scale_both", and "stretch_both"

    return p

def bk_repro_layout(p, num_in_row=1, min_height=360):
    p = bk_legend(p, location='center', orientation = 'horizontal')
    p.legend.visible = False
    p.add_layout(p.legend[0], 'above')

    p.title.text_font_size = '100%'
    p.title.align = 'left'

    p.xaxis.formatter=DatetimeTickFormatter(months=["%b '%y"])
    p.yaxis.axis_label_text_font_size = '100%'
    p.yaxis.major_label_text_font_size = '80%'
    p.yaxis.minor_tick_line_alpha = 0

    p.toolbar.logo = None
    p.toolbar_location = None

    p = add_bokeh_footnote(p)
    if num_in_row > 1:
        p.plot_height = min_height
        p.plot_width = p.plot_height
        p.sizing_mode = 'scale_both'
    else:
        p.plot_width = int(min_height*5/3)
        p.plot_height = min_height
        p.sizing_mode = 'stretch_both'

    #Possible values are "fixed", "scale_width", "scale_height", "scale_both", and "stretch_both"

    return p

def bk_add_event_lines(p, df_int):
    # df_int = df_interventions[df_interventions.state_code.isin([state, 'US'])].groupby('dt').first().reset_index()
    for thisidx in df_int.index:
        if df_int.loc[thisidx, 'social_distancing_direction'] == 'holiday':
            thislinecolor = '#8900a5'
        elif df_int.loc[thisidx, 'social_distancing_direction'] == 'restricting':
            thislinecolor = '#973200'
        elif df_int.loc[thisidx, 'social_distancing_direction'] == 'easing':
            thislinecolor = '#178400'
        p.add_layout(Span(location=df_int.loc[thisidx, 'dt'],
                          dimension='height',
                          line_color='black', #thislinecolor,
                          line_dash='solid',
                          line_alpha=.3,
                          line_width=2
                          )
                     )
    return p

def bk_bar_and_line_chart(bar_series, bar_name='bar', bar_color='#008fd5',
                          line_series=False, line_name='line', line_color='#fc4f30',
                          title='', subtitle='', yformat='{:.1%}',
                          bar2_series=None, bar2_name='bar2', bar2_color='#e5ae38'):

    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=title, subtitle=subtitle)

    p.yaxis.formatter = NumeralTickFormatter(format="0a")

    if isinstance(bar2_series, pd.Series):
        source = pd.concat([line_series, bar2_series, bar_series], axis=1).reset_index()
        source.columns = ['dt', line_name, bar2_name, bar_name]
        p.vbar_stack(stackers=[bar2_name, bar_name],
                     x='dt',
                     color=[bar2_color, bar_color],
                     source=source, width=pd.Timedelta(days=1) * .5,
                     legend_label=[bar2_name, bar_name],
                     name=[bar2_name, bar_name]
                     )
        p.y_range = Range1d(0, max(bar_series.max(), line_series.max(), bar2_series.max()))
    else:
        source = pd.concat([line_series, bar_series], axis=1).reset_index()
        source.columns = ['dt', line_name, bar_name]
        p.vbar(x='dt', top=bar_name,
               source=source, color=bar_color, width=pd.Timedelta(days=1) * .5,
               legend_label=bar_name,
               name=bar_name
               )
        p.y_range = Range1d(0, max(bar_series.max(), line_series.max()))

    p.line(x='dt', y=line_name, source=source, color=line_color, width=4, legend_label=line_name, name=line_name)

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Name', '$name'),
            ('Value', '@$name{0,0}')
        ],
        formatters={'@dt': 'datetime'}
    ))


    return p

def bk_rt_confid(model_dict, simplify=True):
    df_rt = model_dict['df_rts_conf'][['weighted_average']].unstack('metric')
    rt_name = df_rt.columns.levels[0][0]
    df_rt = df_rt[rt_name].dropna(how='all').reset_index()

    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=model_dict['region_name'], subtitle="Reproduction Rate (Rᵗ) Estimate")

    patch = p.varea(x='dt', y1='rt_l68', y2='rt_u68', source=df_rt,
                    color='#E39D22', alpha=0.75, legend_label='68% Confidence Interval', level='glyph')
    # if not simplify:
    patch2 = p.varea(x='dt', y1='rt_l95', y2='rt_u95', source=df_rt,
                     color='#E39D22', alpha=0.25, legend_label='95% Confidence Interval', level='glyph')

    bg_upper = p.varea(x=[df_rt.dt.min(), df_rt.dt.max()],
                       y1=[1.0, 1.0], y2=[df_rt.rt_u95.max(), df_rt.rt_u95.max()],
                       color='red', level='underlay', alpha=0.1
                       )
    bg_lower = p.varea(x=[df_rt.dt.min(), df_rt.dt.max()],
                       y1=[0, 0], y2=[1.0, 1.0],
                       color='blue', level='underlay', alpha=0.1
                       )

    p.line(x='dt', y='rt', source=df_rt, color='#008FD5', width=4,
           legend_label='Reproduction Rate, Rᵗ', level='glyph')

    p.add_layout(Span(location=1.0,
                      dimension='width',
                      line_color='black',  # thislinecolor,
                      line_dash='dashed',
                      line_alpha=.7,
                      line_width=2
                      )
                 )

    if not simplify:
        p.add_layout(Label(
            x=df_rt.dt.mean(), y=1.5, y_units='data', text='Rᵗ > 1: Epidemic Worsening',
            text_color='black', text_alpha=0.4, text_font_size='2vw', text_align='center'))
        p.add_layout(Label(
            x=df_rt.dt.mean(), y=0.1, y_units='data', text='Rᵗ < 1: Epidemic Improving',
            text_color='black', text_alpha=0.4, text_font_size='2vw', text_align='center'))

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Reproduction Rate, Rᵗ', '@rt{0.00}'),
            ('68% Confidence Interval', '@rt_l68{0.00} - @rt_u68{0.00}'),
            ('95% Confidence Interval', '@rt_l95{0.00} - @rt_u95{0.00}')
        ],
        formatters={'@dt': 'datetime'},
        mode='vline'
    ))

    p.y_range = Range1d(0,2) #(0, df_rt.rt_u68.max())

    return p

def bk_population_share(model_dict):
    df_agg = model_dict['df_agg']
    col_names = ['susceptible', 'deaths', 'exposed', 'hospitalized', 'infectious', 'recovered_unvaccinated',
                 'vaccinated_prev_infected', 'vaccinated_never_infected']
    legend_names = ['Susceptible', 'Deaths', 'Exposed',
                    'Infectious', 'Hospitalizations', 'Recoveries, Unvaccinated',
                    'Vaccinated, Previously Infected', 'Vaccinated, Never Infected']
    df_chart = df_agg[col_names].dropna(how='all')
    df_chart = df_chart.clip(lower=0)
    df_chart = df_chart.iloc[8:].reset_index()

    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=model_dict['region_name'], subtitle='Forecast Population Overview')

    p.varea_stack(col_names,
                  x='dt', source=df_chart,
                  color=('#a6cee3', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#1f78b4', '#b2df8a', '#33a02c'),
                  legend_label=legend_names,
                  alpha=0.7
                  )

    p.vline_stack('susceptible',
                  x='dt', source=df_chart,
                  width=0
                  )

    p.y_range = Range1d(0, df_chart.sum(axis=1).max())
    p.yaxis.formatter = NumeralTickFormatter(format="0a")
    p.yaxis.axis_label = 'Population'

    # Setting the second y axis range name and range
    p.extra_y_ranges = {"foo": Range1d(start=0, end=1)}

    p.add_layout(Span(location=pd.Timestamp.today(),
                     dimension='height',
                     line_color='black',  # thislinecolor,
                     line_dash='dashed',
                     line_alpha=.7,
                     line_width=2
                     )
                )
    p.add_layout(Label(
        x=pd.Timestamp.today(), y=0, y_units='data', text='Today',
        text_color='black', text_alpha=0.4, text_font_size='2vw', text_align='center',
        text_baseline='bottom'
    ))

    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="foo", #axis_label="% of Population",
                            major_tick_out=8, major_tick_line_alpha=.9,
                            minor_tick_in=4, minor_tick_line_alpha=.9,
                            formatter=NumeralTickFormatter(format="0%")), 'right')

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Susceptible Population', '@susceptible{0,0}'),
            ('Deaths', '@deaths{0,0}'),
            ('Exposures', '@exposed{0,0}'),
            ('Infectious Population', '@infectious{0,0}'),
            ('Hospitalizations', '@hospitalized{0,0}'),
            # ('Recoveries', '@recovered{0,0}'),
            ('Recoveries, Unvaccinated', '@recovered_unvaccinated{0,0}'),
            ('Vaccinated, Previously Infected', '@vaccinated_prev_infected{0,0}'),
            ('Vaccinated, Never Infected', '@vaccinated_never_infected{0,0}')
        ],
        formatters={'@dt': 'datetime'},
        mode='vline'
    ))

    return p

def bk_postestshare(model_dict):
    df_chart = model_dict['df_hist'][['cases_daily', 'pos_neg_tests_daily']].clip(lower=0)
    df_chart['neg_tests_daily'] = (df_chart['pos_neg_tests_daily'] - df_chart['cases_daily']).clip(lower=0)
    df_chart = df_chart.div(df_chart[['cases_daily','pos_neg_tests_daily']].max(axis=1), axis=0).dropna(how='all')
    df_chart['sevendayavg'] = df_chart['cases_daily'].mask(df_chart['cases_daily'] >= 1.0
                                                           ).rolling(7, min_periods=1).mean()

    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=model_dict['region_name'], subtitle='COVID-19 Positivity Rate')

    p.varea_stack(['cases_daily', 'neg_tests_daily'],
                  x='dt', source=df_chart,
                  color=['#e5ae38', '#008fd5'],
                  legend_label=['% Positive Tests', '% Negative Tests'],
                  alpha=0.7
                  )

    p.line(x='dt', y='sevendayavg', source=df_chart, color='#fc4f30', width=4,
           legend_label='Positivity Rate (7-Day Rolling Average)')

    p.y_range = Range1d(0, min(1, df_chart['sevendayavg'].max() * 1.1))
    p.yaxis.formatter = NumeralTickFormatter(format="0%")
    p.yaxis.axis_label = '% of Daily Tests'

    # Adding the second axis to the plot.
    p.extra_y_ranges = {"foo": p.y_range}
    p.add_layout(LinearAxis(y_range_name="foo", #axis_label=p.yaxis.axis_label,
                            major_tick_out=5, major_tick_line_alpha=.9,
                            minor_tick_in=4, minor_tick_line_alpha=.9,
                            formatter=p.yaxis.formatter), 'right')

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Daily Positivity Rate', '@cases_daily{0.0%}'),
            ('7-Day Rolling Average Positivity Rate', '@sevendayavg{0.0%}')
        ],
        formatters={'@dt': 'datetime'},
        mode='vline'
    ))

    return p

def bk_positivetests(model_dict):
    p = bk_bar_and_line_chart(bar_series=model_dict['df_hist']['cases_daily'].dropna(how='all'),
                       bar_name='Positive Tests', bar_color='#e5ae38',
                       line_series=model_dict['df_hist']['cases_daily'].rolling(7, min_periods=1).mean(),
                       line_name='7-Day Rolling Average', yformat='{:0,.0f}',
                       title=model_dict['region_name'], subtitle='Positive COVID-19 Tests Per Day'
                       )
    return p

def bk_totaltests(model_dict):
    p = bk_bar_and_line_chart(bar_series=model_dict['df_hist']['pos_neg_tests_daily'].dropna(how='all'),
                          bar_name='Negative Tests',
                          line_series=model_dict['df_hist']['pos_neg_tests_daily'].rolling(7, min_periods=1).mean(),
                          line_name='All Tests (7-Day Avg)', yformat='{:0,.0f}',
                          title=model_dict['region_name'], subtitle='COVID-19 Tests Per Day',
                          bar2_series=model_dict['df_hist']['cases_daily'], bar2_name='Positive Tests'
                          )
    return p

def bk_detection_rt(df_agg, model_dict):
    df_chart = model_dict['df_hist']['cases_daily'].rolling(7).mean().div(df_agg['exposed_daily']).dropna()
    df_chart = df_chart.reset_index()
    df_chart.columns = ['dt','detection_rt']
    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=model_dict['region_name'], subtitle='COVID-19 Daily Infection Detection Rate')

    p.yaxis.formatter = NumeralTickFormatter(format="0%")

    p.line(x='dt', y='detection_rt', source=df_chart, width=4, legend_label='Daily Infection Detection Rate')

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Detection Rate', '@detection_rt')
        ],
        formatters={'@dt': 'datetime'}
    ))

    return p

def bk_googmvmt(model_dict):
    df_chart = model_dict['df_mvmt']
    col_names = ['retail_and_recreation_percent_change_from_baseline',
         'grocery_and_pharmacy_percent_change_from_baseline',
         'parks_percent_change_from_baseline',
         'transit_stations_percent_change_from_baseline',
         'workplaces_percent_change_from_baseline',
         'residential_percent_change_from_baseline']
    df_chart = df_chart[col_names]
    df_chart = df_chart.interpolate(limit_area='inside').rolling(7).mean().div(100.0).dropna()
    df_chart = df_chart.reset_index()

    labels = [x[:-29].title().replace('_', ' ') for x in col_names]
    colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']

    p = figure(x_axis_type="datetime",
               tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')
    p = bk_title(p, title=model_dict['region_name'], subtitle='Google Movement Data (Rolling 7-day Average)')

    p.yaxis.formatter = NumeralTickFormatter(format="0%")

    for colidx in range(len(col_names)):
        line_name = labels[colidx]
        p.line(x='dt', y=col_names[colidx], source=df_chart, width=4, legend_label=line_name,
               name=line_name, color=colors[colidx])

    p.add_tools(HoverTool(
        tooltips=[
            ('Date', '@dt{%F}'),
            ('Name', '$name'),
            ('Value', '@$name')
        ],
        formatters={'@dt': 'datetime'}
    ))

    return p

def bk_compare_exposures(df_census, df_fore_allstates):
    # df_chart = df_fore_allstates.stack().unstack('metric')['exposed_daily'].unstack(1)
    s_pop = df_census.loc[(df_census.SUMLEV == 40)].set_index('state')['pop2019']
    df_exposed_daily_per100k = df_fore_allstates.stack().unstack('metric')[
        'exposed_daily'].unstack(1).div(s_pop).mul(1e5)
    df_chart = df_exposed_daily_per100k
    names_sub = df_chart.columns.to_list()
    data = df_chart.reset_index()
    source = ColumnDataSource(data)
    # create CDS for filtered sources
    state1 = 'NY'
    filt_data1 = data[['dt', state1]].rename(columns={state1: 'cases'})
    src2 = ColumnDataSource(filt_data1)

    state2 = 'CA'
    filt_data2 = data[['dt', state2]].rename(columns={state2: 'cases'})
    src3 = ColumnDataSource(filt_data2)

    p1 = figure(x_axis_type='datetime',
                tools='ypan,zoom_in,zoom_out,reset,save',
                title='Model-Estimated Daily New COVID-19 Infections Per 100,000 Residents',
                y_range=Range1d(start=0, end=filt_data1.cases.max() + 50, bounds=(0, None)),
                sizing_mode="scale_width", plot_height=400
                )
    hover_tool = HoverTool(tooltips=
                           [('Date', '@dt{%x}'),
                            ('State', '$name'),
                            ('New Infections Per 100k Residents', '@cases{0}')],
                           formatters={'@dt': 'datetime'}
                           )
    p1.add_tools(hover_tool)

    p1.line(x='dt', y='cases', source=src2,
            legend_label="State 1",
            name="State 1", line_color='blue',
            line_width=3, line_alpha=.8)

    # set the second y-axis and use that with our second line
    # p1.extra_y_ranges = {"y2": Range1d(start=0, end=filt_data2.cases.max()+50)}
    p1.extra_y_ranges = {"y2": p1.y_range}
    p1.add_layout(LinearAxis(y_range_name="y2",
                             formatter=NumeralTickFormatter(format="0a")
                             ),
                  'right')
    p1.line(x='dt', y='cases', source=src3,
            legend_label="State 2",
            name="State 2", line_color='orange',
            line_width=3, line_alpha=.8, y_range_name="y2")

    p1.yaxis[0].axis_label = 'New Infections Per 100k People'
    p1.yaxis[1].axis_label = p1.yaxis[0].axis_label
    p1.yaxis.formatter = NumeralTickFormatter(format="0a")
    p1.legend.location = "top_left"
    p1.xaxis.axis_label = 'Date'

    p1.xaxis.formatter = DatetimeTickFormatter(days='%b %d', months='%b %d')
    # this javascript snippet is the callback when either select is changed

    # y_range.end = parseInt(y[y.length - 1]+50);
    code = """
    var c = cb_obj.value;
    var y = s1.data[c];
    var other_y = s3.data['cases'];
    const y_nonan = y.filter(function (value) {
        return !Number.isNaN(value);
    });
    var y_max = Math.max(...y_nonan);
    const other_y_nonan = other_y.filter(function (value) {
        return !Number.isNaN(value);
    });
    var other_y_max = Math.max(...other_y_nonan);
    var both_ys_max = Math.max(other_y_max, y_max);
    s2.data['cases'] = y;
    y_range.start = 0;
    y_range.end = parseInt(both_ys_max*1.05);
    s2.change.emit();
    """
    callback1 = CustomJS(args=dict(s1=source,
                                   s2=src2,
                                   s3=src3,
                                   y_range=p1.y_range
                                   ), code=code)
    callback2 = CustomJS(args=dict(s1=source,
                                   s2=src3,
                                   s3=src2,
                                   y_range=p1.y_range
                                   ), code=code)

    select1 = Select(title="State 1:", value=state1, options=names_sub)
    select1.js_on_change('value', callback1)
    select2 = Select(title="State 2:", value=state2, options=names_sub)
    select2.js_on_change('value', callback2)
    # btn = Button(label='Update')

    p1.add_layout(Span(location=pd.Timestamp.today(),
                      dimension='height',
                      line_color='black',  # thislinecolor,
                      line_dash='dashed',
                      line_alpha=.7,
                      line_width=2
                      )
                 )
    p1.add_layout(Label(
        x=pd.Timestamp.today(), y=0, y_units='data', text='Today',
        text_color='black', text_alpha=0.4, text_font_size='2vw', text_align='center',
        text_baseline='bottom'
    ))

    p1 = bk_legend(p1)
    p1 = add_bokeh_footnote(p1)
    p1.legend.orientation = 'horizontal'
    p1.add_layout(p1.legend[0], 'above')

    curdoc().theme = bk_theme
    layout = column(row(select1, select2), row(p1), sizing_mode='scale_width')
    return layout