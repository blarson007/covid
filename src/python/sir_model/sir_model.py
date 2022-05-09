import pandas as pd


def get_prob_contact_spread(cases_avg, next_cases_avg, immune=0, total_pop=332403650, prob_recovery=.14285714857143):
    """
    SIR Model implementation, where we are solving for the probability of contact times
    the probability of spread.
    """
    inner = cases_avg / total_pop * (total_pop - immune)
    return next_cases_avg / inner + prob_recovery * cases_avg / inner - cases_avg / inner


def prob_contact(prob_contact_spread, variant_score):
    return prob_contact_spread / variant_score


def compute_sir():
    # Case data from NY Times dataset
    case_data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/us.csv'
    df = pd.read_csv(case_data_url, parse_dates=['date'])

    # Start in April of 2020
    df = df[df.date > '2020-04-01']

    # We don't need these fields for our analysis
    df.drop(columns=['geoid', 'cases_avg_per_100k', 'deaths', 'deaths_avg', 'deaths_avg_per_100k'], inplace=True)

    # constants
    total_pop = 332403650  # US population
    prob_recovery = .14285714857143  # 1/7 probability rate, or will likely take 7 days to recover

    # This is the number of infected individuals at time period t + 1
    df['next_cases_avg'] = -(df.cases_avg.diff(periods=-1)) + df.cases_avg

    # Compute our new Susceptible using vaccination data
    # Rules:
    # - Vaccines will be X% effective
    # - Vaccine efficacy will last for X months for full vaccinations
    # - Vaccine efficacy will last for X months for boosters
    # - Vaccine efficacy will last for X months for partial vaccinations
    # - Immunity will last for X months for those who have had covid

    # Let's define the susceptible population to exclude people who have had covid in the last 60 days
    df['immune_period'] = df.cases.rolling(60, min_periods=1).sum()

    # pull in vaccination data set
    owd_url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    owd_df = pd.read_csv(owd_url, parse_dates=['date'])

    # Limit the data to only United States
    us_owd = owd_df[owd_df.location == 'United States']

    # Only include features that are related to vaccines
    vaccine_df = us_owd[[
        'date',
        'total_vaccinations',
        'people_vaccinated',
        'people_fully_vaccinated',
        'total_boosters',
        'new_vaccinations',
        'new_vaccinations_smoothed',
        'total_vaccinations_per_hundred',
        'people_vaccinated_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'new_people_vaccinated_smoothed',
        'new_people_vaccinated_smoothed_per_hundred'
    ]]

    # Build features to identify new vaccines since the last time period
    vaccine_df['new_people_fully_vaccinated'] = -(vaccine_df.people_fully_vaccinated.shift(1) - vaccine_df.people_fully_vaccinated)
    vaccine_df['new_people_fully_vaccinated_smoothed'] = vaccine_df.new_people_fully_vaccinated.rolling(7, min_periods=1).mean()

    # Build features to identify new boosters since the last time period
    vaccine_df['new_boosters'] = -(vaccine_df.total_boosters.shift(1) - vaccine_df.total_boosters)
    vaccine_df['new_boosters_smoothed'] = vaccine_df.new_boosters.rolling(7, min_periods=1).mean()

    # Still have a gap - partially vaccinated individuals
    # Per https://ourworldindata.org/covid-vaccinations, at this time 77% of United States population
    # has been vaccinated: 66% fully vaccinated, and 10% partially vaccinated
    # We will compute the difference as a proportion of fully vaccinated, then apply this difference evenly across
    # all instances of full vaccinations in the data set
    prop_partial_vax = ((total_pop * .77) - (total_pop * .66)) / (total_pop * .77)

    # Build features to identify partially vaccinated people since the last time period
    vaccine_df['people_partially_vaccinated'] = vaccine_df.new_people_fully_vaccinated * prop_partial_vax
    vaccine_df['people_partially_vaccinated_smoothed'] = vaccine_df.people_partially_vaccinated.rolling(7, min_periods=1).mean()

    # merge vaccination data set to main dataframe and fill in empty values
    merge_df = pd.merge(df, vaccine_df, how='left', left_on='date', right_on='date')
    merge_df = merge_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

    # compute Susceptible population
    vax_effectiveness = .75  # 75% effectiveness
    vax_duration = 182  # lasts for 6 months
    booster_duration = 182  # also lasts for 6 months
    partial_vax_duration = 90  # Partially vaccinated last for 3 months

    # Multiply the vaccine effectiveness by the number of vaccinated individuals over a rolling 6 month window
    merge_df['vax_coverage'] = vax_effectiveness * merge_df.new_people_fully_vaccinated_smoothed.rolling(vax_duration, min_periods=1).sum()

    # Multiply the vaccine effectiveness by the number of boosted individuals over a rolling 6 month window
    merge_df['booster_coverage'] = vax_effectiveness * merge_df.new_boosters.rolling(booster_duration, min_periods=1).sum()

    # Multiply the vaccine effectiveness by the number of partially vaccinated individuals over a rolling 3 month window
    merge_df['partial_coverage'] = vax_effectiveness * merge_df.people_partially_vaccinated.rolling(partial_vax_duration, min_periods=1).sum()

    # Immune is defined by those who are covered by vaccines, added to the number of currently or recently infected
    merge_df['immune'] = merge_df.vax_coverage + merge_df.partial_coverage + merge_df.booster_coverage + merge_df.immune_period

    # Apply the modified SIR model to the case data to compute our probability of contact times the probability of spread: prob_contact_spread
    merge_df['prob_contact_spread'] = merge_df.apply(
        lambda x: get_prob_contact_spread(x.cases_avg, x.next_cases_avg, x.immune), axis=1
    )

    # R Zero: probability of contact times probability of spread, divided by probability of recovery
    merge_df['r_zero'] = merge_df.prob_contact_spread / prob_recovery

    # 14 day moving average for a smoother visualization. Shows trends instead of just being noisy.
    merge_df['r_zero_ma'] = merge_df.r_zero.rolling(14, min_periods=1).mean()

    # Read in the probability of spread that was computed from variant data
    prob_spread_df = pd.read_csv('../variant/output/prob_spread.csv', parse_dates=['date'])

    # Merge in the probability of spread data, which is time series data
    # It doesn't line up perfectly, since there isn't a single value for every day
    # So we'll merge to the closest value available
    full_df = pd.merge_asof(
        merge_df,
        prob_spread_df,
        on='date',
        tolerance=pd.Timedelta('6d'))

    # Finally, we can extract the probability of contact from the R Zero value
    full_df['prob_contact'] = full_df.apply(lambda x: prob_contact(x.prob_contact_spread, x.score), axis=1)

    # 7 day moving average of probability of contact
    full_df['prob_contact_smoothed'] = full_df.prob_contact.rolling(7, min_periods=1).mean()

    # Output the results to a csv
    full_df.to_csv('output/full_df.csv')
