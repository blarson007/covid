import json
import pandas as pd
import re


def wrap_countries(country_dict):
    return json.dumps(
        {
            'countries': country_dict
        }
    )


def build_variant_dict(variants):
    variant_dict = {}
    for variant in variants:
        if '(' in variant:
            base_string = variant[variant.find('(') + 1: variant.find(')')]
            variant_dict[variant] = base_string.split('+')
        else:
            variant_dict[variant] = []
    return variant_dict


def match_variant(lineage_val, variant_dict):
    for key, val in variant_dict.items():
        for regex in val:
            if re.search(r'%s' % regex, lineage_val):
                return regex
    return ''


def subvariants(variant_dict):
    items = []
    for key, val in variant_dict.items():
        for subvariant in val:
            items.append({
                'variant': key,
                'subvariant': subvariant
            })
    return pd.DataFrame(items)


def get_grouped_variant(lineage_val):
    delta_regex = 'AY.*'
    omicron_ba1_regex = 'BA.1.*'
    omicron_ba2_regex = 'BA.*'

    if pd.isna(lineage_val):
        return 'Alpha'
    if re.search(r'%s' % delta_regex, lineage_val):
        return 'Delta'
    if re.search(r'%s' % omicron_ba1_regex, lineage_val):
        return 'Omicron BA1'
    if re.search(r'%s' % omicron_ba2_regex, lineage_val):
        return 'Omicron BA2'
    return 'Alpha'


def score_variant(variant, percentage):
    alpha = .4
    delta = 1 - (1 - alpha) * .5
    omicron_ba1 = 1 - (1 - delta) * .5
    omicron_ba2 = 1 - (1 - omicron_ba1) * .5

    if variant == 'Delta':
        return percentage * delta / 100
    if variant == 'Omicron BA1':
        return percentage * omicron_ba1 / 100
    if variant == 'Omicron BA2':
        return percentage * omicron_ba2 / 100
    return percentage * alpha / 100


def build_score():
    data = pd.read_json('~/Downloads/gisaid_variants_statistics_2022_04_11_0054/gisaid_variants_statistics.json')

    # The json is malformed, so we're going to need to manipulate it quite a bit
    # We'll start with converting the nested objects into a named list
    data['countries'] = data.stats.apply(wrap_countries)

    # The data is sparse before this date
    data = data[data.index > '2021-02-01']

    result_df = pd.DataFrame()

    for index, row in data.iterrows():
        value = json.loads(row.countries)
        # Only concerned with USA data at this time
        if 'USA' not in value['countries']:
            continue

        usa = value['countries']['USA']

        # Get the lineage data in json format
        lineage_df = pd.json_normalize(
            usa,
            record_path=['submissions_per_lineage']
        )

        # Rename the key/value columns into something that makes sense to us
        lineage_df.rename(
            columns={
                'count': 'lineage_count',
                'value': 'lineage'
            }, inplace=True
        )

        # If we don't have any lineage data for the date/country, keep going
        if len(lineage_df.index) == 0:
            continue

        # Get the variant data in json format
        variant_df = pd.json_normalize(
            usa,
            record_path=['submissions_per_variant'],
            meta=[
                'submissions'
            ]
        )

        # Rename the key/value columns into something that makes sense to us
        variant_df.rename(
            columns={
                'count': 'variant_count',
                'value': 'variant',
                'submissions': 'submission_count'
            }, inplace=True
        )

        vals = variant_df.variant.values.tolist()

        # Parse the variant string into subvariant regular expressions
        variant_dict = build_variant_dict(vals)
        # Convert from dictionary into dataframe so we can merge it with our variants
        subvariant_df = subvariants(variant_dict)
        if len(subvariant_df.index) == 0:
            continue

        # Merging variant dataframe with subvariant dataframe so we can have a unique row
        # per subvariant, as determined by the variant string
        # For example: "VOC Delta GK (B.1.617.2+AY.*) first detected in India"
        # contains two subvariants: B.1.617.2 and AY.*
        variant_df = variant_df.merge(subvariant_df, left_on='variant', right_on='variant')

        # Prepare the data so that the variants can be merged with lineage data
        # We do this by looking in the variant data for a value that matches the regular expression
        # that was extracted from the subvariant data above
        # This will give us something to join on
        lineage_df['lineage_variant'] = lineage_df.apply(lambda x: match_variant(x.lineage, variant_dict), axis=1)

        # Now that we have a variant string to merge on, we can finally perform the merge of variant with lineage
        full_variant_df = variant_df.merge(lineage_df, how='outer', left_on='subvariant', right_on='lineage_variant')
        full_variant_df['date'] = index

        # The "grouped_variant" is a list of variants that we want to call special attention to.
        # For our purposes, any non-Delta and non-Omicron variants are treated the same as Alpha
        full_variant_df['grouped_variant'] = full_variant_df.lineage.apply(get_grouped_variant)

        result_df = result_df.append(full_variant_df)

    # Remove all 'Unassigned' lineage, since we can't do anything with that
    result_df = result_df[result_df.lineage != 'Unassigned']

    # Group all lineage counts based on date as well as the variant list that we care about
    grouped = result_df.groupby(['date', 'grouped_variant'])['lineage_count'].sum().reset_index()

    # Remove any entries that have no lineage counts
    grouped = grouped[grouped.lineage_count > 0]

    # For a given date, determine the percentage of lineage_count that are accounted for by any given variant
    grouped['percentage'] = grouped.groupby('date')['lineage_count'].apply(lambda x: 100 * x / float(x.sum()))

    # Apply a score to each date, based on the percentage of lineage count for that variant
    grouped['score'] = grouped.apply(lambda x: score_variant(x.grouped_variant, x.percentage), axis=1)
    date_score = grouped.groupby('date')['score'].sum().reset_index()

    date_score.to_csv('output/prob_spread.csv')
    # grouped.to_csv('output/variant.csv')
