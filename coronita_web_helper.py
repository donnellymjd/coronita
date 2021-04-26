import pandas as pd

dict_ch_defs = {}

dict_ch_defs['ch_rt_confid'] = """Reproduction Rate with Confidence Intervals
: The reproduction rate (sometimes referred to as the Rt, R0, or "R-naught") is the average number of people that each infected person transmits COVID-19 to. If the reproduction rate is below 1.0, then the epidemic will fizzle out. If the reproduction rate is above 1.0, then the epidemic is spreading exponentially. More information about how this chart was calculated is available on the methodology page.
"""

dict_ch_defs['ch_rts'] = """Individual Reproduction Rate Estimates
: Underlying the reproduction rate (Rt) estimate that the forecasting model uses are several reproduction rates based on various data, like cases, positivity rates, deaths, and hospitalizations. Each type of data has pros and cons. For example, case data (i.e. positive tests) can tell us of recent spikes in infections, but case data are not necessarily representative because they are affected by the number of tests conducted. On the other hand, death data are generally fairly representative but represent a long delay in changes of infection rates because people who die from COVID-19 on average pass away several weeks after infection.
"""

dict_ch_defs['ch_statemap'] = """New COVID-19 Cases Per 100k Residents Over Last 14 Days
: Newly reported COVID-19 cases per 100,000 residents over the most recent two weeks. The top end of the color scale is set at the 90th percentile of all US counties. This chart is based on reported data rather than modeled data. As a result, states with higher testing rates will appear worse off than states with lower testing rates.
"""

dict_ch_defs['ch_statemap_total'] = """Cases Per 100,000 Residents
: Reported cumulative COVID-19 cases per 100,000 residents. Many of the cases reported are no longer infectious as the vast majority of people who become infected with COVID-19 recover. The top end of the scale is set at the 90th percentile of all US counties.
"""

dict_ch_defs['ch_googmvmt'] = """Google Movement Data
: Through its [Community Mobility Reports](https://www.google.com/covid19/mobility/), Google has made public aggregate data that represent "movement trends over time by geography, across different categories of places such as retail and recreation, groceries and pharmacies, parks, transit stations, workplaces, and residential." These are percent changes from baseline (roughly the average of visits in February 2020.) Notice the spike in grocery and pharmacy visits in mid-March in most states.
"""

dict_ch_defs['ch_exposed_infectious'] = """Simultaneous Infections Forecast
: The model is able to estimate the number of people with active infections in a region over time. The exposed population refers to people who have been infected very recently but are pre-symptomatic and less likely to be able to spread COVID-19. The Infectious Population is an estimate of the number of people who are currently experiencing symptoms and are able to spread COVID-19.
"""

dict_ch_defs['ch_daily_exposures'] = """Daily Exposures Forecast
: This chart shows the number of *new infections* estimated by the model every day. We show it alongside the number of new positive test results to show the estimated undercounting of new cases. We call this the exposed population because refers to people who have been infected very recently but are pre-symptomatic and less likely to be able to spread COVID-19.
"""

dict_ch_defs['ch_positivetests'] = """Positive COVID-19 Tests Per Day
: This chart shows the number of *daily new infections* detected by RT-PCR testing, the kind of tests that look for active infections (rather than antibody tests that look for evidence of past infections). The chart also plots the 7-day rolling average as a red line, which takes the simple mean of each day and the previous six days, in order to give a better sense of changes in the trend of new cases. Some states do not report this data every day, resulting in big spikes and dips. Many states also have consistently higher results on some days of the week, likely due to testing centers and lab opening hours. The 7-day rolling average line should give a more accurate sense of changes in trend.
"""

dict_ch_defs['ch_totaltests'] = """Total COVID-19 Tests Per Day
: This chart shows positive and negative RT-PCR daily test results stacked on top of each other, meaning the top of each bar represents the total number of test results reported each day. The chart also plots the 7-day rolling average as a red line, which takes the simple mean of each day and the previous six days, in order to give a better sense of changes in the trend of new cases. Some states do not report this data every day, resulting in big spikes and dips. Many states also have consistently higher results on some days of the week, likely due to testing centers and lab opening hours. The 7-day rolling average line should give a more accurate sense of changes in trend.
"""

dict_ch_defs['ch_postestshare'] = """Daily Positive COVID-19 Test Share (Positivity Rate)
: Similar to the previous chart, this chart shows positive and negative RT-PCR daily test results stacked on top of each other, but each day's results are normalized so the positive and negative results add up to 100%. The chart also plots the 7-day rolling average as a red line, which takes the simple mean of each day and the previous six days, in order to give a better sense of changes in the trend of new cases. Some states do not report this data every day, resulting in big spikes and dips. Many states also have consistently higher results on some days of the week, likely due to testing centers and lab opening hours. The 7-day rolling average line should give a more accurate sense of changes in trend.
"""

dict_ch_defs['ch_hosp'] = """Hospitalization and Deaths Forecast
: The core model output is an estimate of the number of COVID-19 hospitalizations and deaths. More information on the methodology of this model is available on the [model methodology page](http://www.michaeldonnel.ly/covid19/methodology/). Concurrent hospitalizations refers to the number of COVID-19 cases estimated/forecast to be in the hospital on any given day. Deaths represents the model's estimate for the total number of deaths to have occurred by a given date. ICU cases are treated as a subset of hospitalizations and ventilations are treated as a subset of ICU cases.
"""

dict_ch_defs['ch_hosp_concur'] = """Hospitalization and Deaths Forecast
: The core model output is an estimate of the number of COVID-19 hospitalizations. More information on the methodology of this model is available on the [model methodology page](http://www.michaeldonnel.ly/covid19/methodology/). Concurrent hospitalizations refers to the number of COVID-19 cases estimated/forecast to be in the hospital on any given day. ICU cases are treated as a subset of hospitalizations and ventilations are treated as a subset of ICU cases.
"""


dict_ch_defs['ch_deaths_tot'] = """Total Deaths Forecast
: The core model output is an estimate of the cumulative number of COVID-19 deaths. More information on the methodology of this model is available on the [model methodology page](http://www.michaeldonnel.ly/covid19/methodology/). Deaths represents the model's estimate for the total number of deaths to have occurred by a given date. 
"""

dict_ch_defs['ch_population_share'] = """Population Overview Forecast
: The forecasting model also outputs estimates of the percentage of the overall population that has a current COVID-19 infection, recovered from a past infection, died from COVID-19, or is currently susceptible because they have not been previously infected. (Currently, since no vaccine exists, the entire uninfected population is represented as susceptible.) More information on the methodology of this model is available on the [model methodology page](http://www.michaeldonnel.ly/covid19/methodology/).
"""

dict_ch_defs['ch_cumul_infections'] = """Cumulative Infections Forecast
: This chart shows the number of *total COVID-19 infections* estimated by the model over time. We show it alongside the number of total positive test results to show the estimated undercounting of cases. 
"""

dict_ch_defs['ch_hosp_admits'] = """Daily Hospital Admissions Forecast
: This chart shows the number of *new hospital admissions with COVID-19 infections* estimated by the model over time. Hospital admissions differ from the earlier concurrent hospitalizations chart. Concurrent COVID-19 hospitalizations are determined by the sum of earlier admissions minus the sum of earlier discharges and deaths. We show it alongside the reported number of hospital admissions where available. Many states do not report this data at all and some only report it sporadically.
"""


dict_ch_defs['ch_doubling_rt'] = """Doubling Rate Forecast
: The doubling rate chart can be a little noisy and confusing. This chart shows how quickly the model expects new cases, hospitalizations, and deaths to double over time (shown in days).
"""


dict_ch_defs['ch_daily_deaths'] = """Daily Deaths Forecast
: This chart shows the number of *new deaths from COVID-19* estimated by the model over time.  Generally, reported COVID-19 death data from states is generally reliable. So there is rarely major difference between the model's estimates and the reported data. Occassionally there is a major spike or dip in the reported data as a result of changes in the reporting definitions from states.
"""


dict_ch_defs['ch_detection_rt'] = """Daily Infection Detection Rate
: This chart shows the model estimated daily percentage of new infections that are detected through positive tests (RT-PCR tests that test for active infections). The formula for this detection rate is new reported daily positive test results / the model's estimate of new infections (new infections in the SEIR model are called "exposures"). Some people will be counted twice in the testing data, especially those in a hospital as doctors try to identify the end of an active infection. That double counting along with small expected model errors, especially at times when the underlying active infectious rate in a state is low, can result in detection rates above 100%.
"""

dict_ch_defs['ch_vax_status'] = """Total People Vaccinated Forecast
: This chart shows the forecasted total number of people with completed vaccinations over time. It also shows the reported number of people with partially complete and completed vaccinations. Our forecast assumes that all adults, except for those who reported as hesitant to receive the vaccine in the US Census Bureau's Household Pulse Survey, will eventually be vaccinated. The rate of vaccinations is expected to slow as we approach the total number of adults who are willing to get vaccinated.
"""

dict_ch_defs['ch_vax_daily'] = """Daily New Vaccinations Initiated Forecast
: This chart shows the forecasted number of people initiating a COVID-19 vaccination per day. It also shows the reported number for the same metric. Our forecast assumes that all adults, except for those who reported as hesitant to receive the vaccine in the US Census Bureau's Household Pulse Survey, will eventually be vaccinated. The rate of vaccinations is expected to slow as we approach the total number of adults who are willing to get vaccinated.
"""

dict_ch_defs['ch_rt_scen_explanation'] = """Reproduction Rate Breakdown & Forecast
: This chart shows the historical and the forecasted effective reproduction rate (Rt) alongside what the reproduction rate hypothetically would have looked like if there were no immunity to COVID-19. In order to calculate the hypothetical Pre-Immunity/Behavioral Reproduction Rate, we divide the observed effective reproduction rate by the percentage of the overall population without any immunity (aka the susceptible population.) To estimate the susceptible population, we model the # of people previously infected and assume they have acquired immunity after their infection AND we forecast the number of people who have completed a vaccination against COVID-19. To avoid double counting, we assume vaccines are randomly distributed across susceptible and recovered populations. By calculating the hypothetical Pre-Immunity/Behavioral Reproduction Rate, we can see how close we are to the prevailing original reproduction rate (aka the basic Reproduction rate) at the beginning of the crisis. The original rate shows us how transmissible COVID-19 is in each region without mitigation measures and without immunity. Since our ultimate goal is to return to "normal life" without mitigation measures like social distancing, we will need to control the spread of COVID-19 through immunity alone. The forecasted hypothetical Rt brings us back to the original R0 by July 4, 2021, with an additional 20% transmissibility due to variants of concern. (To prevent nonsensical results, we constrain the current VOC R0 to be no less than 2.0 and no greater than 3.0 in any state.) If after that July 4th date, the forecasted effective Rt is below 1, then the region has achieved herd immunity and will eventually see their infection rate drop to very low levels.
"""


