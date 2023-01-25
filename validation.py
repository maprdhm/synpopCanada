import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 7)
matplotlib.rcParams.update({'font.size': 12})


# Load statistics from census 2021
def load_stats_census(path, year):
    file = path + '/census_' + year + '_stats.csv'
    df_pop = pd.read_csv(file, usecols=['area', 'population', 'males', 'females', 'population private dwellings',
                                        'households', 'avg age', 'med age', 'avg hh size', '0-4', '5-9', '10-14',
                                        '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                                        '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94',
                                        '95-99', '100+', 'hh 1p', 'hh 2p', 'hh 3p', 'hh 4p', 'hh 5+p',
                                        '15+ income < 20k$', '15+ income 20k$<= and <60k$',
                                        '15+ income 60k$<= and <100k$', '15+ income >= 100k$', '15+',
                                        'Couples without children', 'Couples with children', 'One-parent-family',
                                        'One-person', 'Other kind of hh'], dtype='object')
    df_pop['area'] = df_pop['area'].astype(str)

    # Treat DAs converted from 2021 to 2016: sort, remove duplicates, sum
    df_pop['area'] = df_pop['area'].str.split('_').map(lambda x: '_'.join(sorted(x)))
    df_pop = df_pop.drop_duplicates(subset=["area", "population", "households"])
    df_pop.loc[df_pop['area'].str.contains('_2016S0512'), 'avg age'] = 'x'
    df_pop.loc[df_pop['area'].str.contains('_2016S0512'), 'med age'] = 'x'
    df_pop.loc[df_pop['area'].str.contains('_2016S0512'), 'avg hh size'] = 'x'
    cols = df_pop.columns.drop('area')
    df_pop[cols] = df_pop[cols].apply(pd.to_numeric, errors='coerce')
    df_pop = df_pop.groupby(df_pop.area, as_index=False).sum()
    df_pop = add_distribution_stats(df_pop)

    return df_pop


# Load statistics from synthetic population 2021
def load_stats_syn_pop(path, year, df_stats_census, scenario):
    file = path + '/syn_pop_' + year + '_stats_'+scenario+'.csv'
    df_pop = pd.read_csv(file, usecols=['area', 'population', 'males', 'females', 'population private dwellings',
                                        'households', 'avg age', 'med age', 'avg hh size', '0-4', '5-9', '10-14',
                                        '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                                        '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94',
                                        '95-99', '100+', 'hh 1p', 'hh 2p', 'hh 3p', 'hh 4p', 'hh 5+p',
                                        '15+ income < 20k$', '15+ income 20k$<= and <60k$',
                                        '15+ income 60k$<= and <100k$', '15+ income >= 100k$', '15+',
                                        'Couples without children', 'Couples with children', 'One-parent-family',
                                        'One-person', 'Other kind of hh', '% realistic individuals'], dtype='object')
    df_pop['area'] = df_pop['area'].astype(str)
    cols = df_pop.columns.drop('area')
    df_pop[cols] = df_pop[cols].apply(pd.to_numeric, errors='coerce')

    # Merge DAs similarly to census data
    merged_das = df_stats_census.loc[df_stats_census['area'].str.contains('_'), 'area']
    for merged_da in merged_das:
        das = merged_da.split('_')
        df_pop.loc[len(df_pop), :] = df_pop.loc[df_pop['area'].isin(das)].sum(axis=0)
        df_pop.loc[len(df_pop) - 1, 'area'] = merged_da
        df_pop.loc[len(df_pop) - 1, '% realistic individuals'] = df_pop.loc[
                                                                     len(df_pop)-1, '% realistic individuals']/len(das)
    for merged_da in merged_das:
        das = merged_da.split('_')
        df_pop.drop(df_pop[df_pop.area.isin(das)].index, inplace=True)

    df_pop = add_distribution_stats(df_pop)

    return df_pop


def load_projections(path, scenario, year):
    file = path + '/projections_pop_17100057.csv'
    proj = pd.read_csv(file, usecols=['REF_DATE', 'GEO', 'DGUID', 'Projection scenario', 'Sex', 'Age group', 'VALUE'],
                       low_memory=False)
    proj = proj.loc[proj['REF_DATE'] == int(year)]
    proj = proj.loc[proj['Projection scenario'] == "Projection scenario " + scenario]
    proj['value'] = proj['VALUE'] * 1000

    df = pd.DataFrame()
    df['area'] = proj['DGUID'].unique()

    for index, row in df.iterrows():
        df_zone = proj.loc[(proj['DGUID'] == row['area'])]
        df.at[index, 'population'] = df_zone.loc[
            (df_zone['Sex'] == 'Both sexes') & (df_zone['Age group'] == 'All ages'), 'value'].iloc[0]
        for i in range(0, 101, 5):
            if i == 100:
                df.at[index, '100+'] = df_zone.loc[
                    (df_zone['Sex'] == 'Both sexes') & (df_zone['Age group'] == '100 years and over'), 'value'].iloc[0]
            else:
                df.at[index, str(i) + '-' + str(i + 4)] = df_zone.loc[(df_zone['Sex'] == 'Both sexes') & (
                            df_zone['Age group'] == str(i) + ' to ' + str(i + 4) + ' years'), 'value'].iloc[0]
        df.at[index, 'males'] = \
        df_zone.loc[(df_zone['Sex'] == 'Males') & (df_zone['Age group'] == 'All ages'), 'value'].iloc[0]
        df.at[index, 'females'] = \
        df_zone.loc[(df_zone['Sex'] == 'Females') & (df_zone['Age group'] == 'All ages'), 'value'].iloc[0]
    df = add_distribution_stats(df)
    return df


def load_estimates(path, year):
    file = path + '/population_estimates_17100009.csv'
    proj = pd.read_csv(file, usecols=['REF_DATE', 'GEO', 'DGUID', 'VALUE'], low_memory=False)
    proj['value'] = proj['VALUE']
    proj = proj.loc[proj['REF_DATE'] == year+'-04']

    df = pd.DataFrame()
    df['area'] = proj['DGUID'].unique()

    for index, row in df.iterrows():
        df_zone = proj.loc[(proj['DGUID'] == row['area'])]
        df.at[index, 'population'] = df_zone['value'].iloc[0]
    return df


# Function to add value labels
def add_labels(x, y, text, shift_x, shift_y):
    for i in range(len(x)):
        t = '{:+.2f}%'.format(float(text[i]))
        plt.text(i + shift_x, y[i] + shift_y, t, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=.8))


def add_distribution_stats(df):
    df['% males'] = df["males"] / df["population"] * 100.0
    df['% females'] = df["females"] / df["population"] * 100.0

    for i in range(0, 101, 5):
        if i == 100:
            df['% 100+'] = df["100+"] / df["population"] * 100.0
        else:
            df['% ' + str(i) + '-' + str(i + 4)] = df[str(i) + '-' + str(i + 4)] / df["population"] * 100.0

    if 'Couples without children' in df.columns:
        df = df.rename(columns={"hh 5+p": "5+", "hh 4p": "4", "hh 3p": "3", "hh 2p": "2", "hh 1p": "1",
                                '15+ income < 20k$': '< 20k $', '15+ income 20k$<= and <60k$': '20k-60k $',
                                '15+ income 60k$<= and <100k$': '60k-100k $', '15+ income >= 100k$': '≥ 100k $'})
        for i in range(1, 6):
            if i == 5:
                df['% 5+'] = df["5+"] / df["households"] * 100.0
            else:
                df['% ' + str(i)] = df[str(i)] / df["households"] * 100.0

        df['% < 20k $'] = df['< 20k $'] / df["15+"] * 100.0
        df['% 20k-60k $'] = df['20k-60k $'] / df["15+"] * 100.0
        df['% 60k-100k $'] = df['60k-100k $'] / df["15+"] * 100.0
        df['% ≥ 100k $'] = df['≥ 100k $'] / df["15+"] * 100.0

        df['% Couples without children'] = df['Couples without children'] / df["households"] * 100.0
        df['% Couples with children'] = df['Couples with children'] / df["households"] * 100.0
        df['% One-parent-family'] = df['One-parent-family'] / df["households"] * 100.0
        df['% One-person'] = df['One-person'] / df["households"] * 100.0
        df['% Other kind of hh'] = df['Other kind of hh'] / df["households"] * 100.0

    return df


def create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, title, df_projections=[], df_diff_projections=[]):
    syn_pop_y = []
    census_y = []
    proj_y = []
    percent_error = []
    for i in x:
        syn_pop_y.append(df_stats_syn_pop_canada["% " + i].iloc[0])
        census_y.append(df_stats_census_canada["% " + i].iloc[0])
        percent_error.append(df_diff["% " + i].iloc[0])
        if i.lower().replace("\n", "") in df_projections:
            proj_y.append(df_projections["% " + i].iloc[0])
        else:
            proj_y.append(0)
    dat = {"Census 2021": census_y, "Synth.pop. 2021": syn_pop_y}
    if i.lower().replace("\n", "") in df_projections:
        dat = {"Census 2021": census_y, "Synth.pop. 2021": syn_pop_y, "Projections 2021": proj_y}
    plotdata = pd.DataFrame(dat, index=x)

    ax = plotdata.plot(kind="bar")
    y = []
    for i in range(len(syn_pop_y)):
        y.append(min(55,max(census_y[i],syn_pop_y[i])))
    add_labels(x, y, percent_error, 0.05, 0.06 * max(max(syn_pop_y), max(census_y)))
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000000), ',')))
    plt.title("Comparison of " + title + " distribution")
    plt.legend(ncol=3)
    plt.xlabel("")
    plt.ylabel("%")
    plt.xticks(rotation=20)
    ax.set_ylim([0, 50])
    plt.tight_layout()
    plt.show()


def create_plot_count(x, df_stats_syn_pop, df_stats_census, df_diff, title, df_projections=[],
                      df_errors_projections=[], df_estimates=[], df_errors_estimates=[]):
    syn_pop_y = []
    census_y = []
    proj_y = []
    est_y = []
    percent_error = []
    proj_y_error=[]
    est_y_error=[]
    for i in x:
        syn_pop_y.append(df_stats_syn_pop[i.lower().replace("\n", "")].iloc[0])
        census_y.append(df_stats_census[i.lower().replace("\n", "")].iloc[0])
        percent_error.append(df_diff[i.lower().replace("\n", "")].iloc[0])
        if i.lower().replace("\n", "") in df_projections:
            proj_y.append(df_projections[i.lower().replace("\n", "")].iloc[0])
            proj_y_error.append(df_errors_projections[i.lower().replace("\n", "")].iloc[0])
            est_y.append((df_estimates[i.lower().replace("\n", "")].iloc[0]))
            est_y_error.append((df_errors_estimates[i.lower().replace("\n", "")].iloc[0]))
        else:
            proj_y.append(0)
            est_y.append(0)

    dat = {"Census 2021": census_y, "Synth.pop. 2021": syn_pop_y}
    if "population" in df_projections:
        dat = {"Census 2021": census_y, "Synth.pop. 2021": syn_pop_y, "Projections 2021": proj_y, "Estimates 2021": est_y}

    plotdata = pd.DataFrame(dat, index=x)

    ax = plotdata.plot(kind="bar")
    add_labels(x, syn_pop_y, percent_error, 0, 0.06 * max(max(syn_pop_y),max(census_y)))
    #add_labels([x[0]], proj_y, proj_y_error, 0.25, 0.05 * max(proj_y))
    #add_labels([x[0]], est_y, est_y_error, 0.5, 0.05 * max(est_y))

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(float(x / 1000000), ',')))
    plt.title("Comparison of " + title + " count - Canada")
    plt.xlabel("")
    plt.legend(loc='upper right', ncol=1)
    plt.ylabel("Count (in millions)")
    plt.xticks(rotation=0)
    ax.set_ylim([0, 1.3 * max(syn_pop_y)])
    plt.tight_layout()
    plt.show()


def compute_errors_Canada(df_stats_census, df_stats_syn_pop, df_projections, df_estimates):
    canada_code = '2016A000011124'
    df_stats_census_canada = df_stats_census.loc[df_stats_census['area'] == canada_code]
    df_stats_syn_pop_canada = df_stats_syn_pop.loc[df_stats_syn_pop['area'] == canada_code]
    df_estimates_canada = df_estimates.loc[df_estimates["area"] == canada_code]

    df_stats_census_canada = df_stats_census_canada.set_index(['area'])
    df_stats_syn_pop_canada = df_stats_syn_pop_canada.set_index(['area'])
    df_projections = df_projections.set_index(['area'])
    df_estimates_canada = df_estimates_canada.set_index(['area'])

    df_diff = abs(df_stats_syn_pop_canada.sub(df_stats_census_canada))
    df_errors = df_diff.div(df_stats_census_canada) * 100.0

    df_diff_projections = abs(df_stats_syn_pop_canada.sub(df_projections))
    df_errors_projections = df_diff_projections.div(df_projections) * 100.0

    df_diff_estimates = abs(df_stats_syn_pop_canada.sub(df_estimates_canada))
    df_errors_estimates = df_diff_estimates.div(df_estimates_canada) * 100.0

    x = ["Population", "Population \nprivate dwellings", "Households"]
    create_plot_count(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "population", df_projections,
                      df_errors_projections, df_estimates_canada, df_errors_estimates)

    print("Canada", round(df_stats_syn_pop_canada['% realistic individuals'].iloc[0], 2), "% of realistic individuals")

    x = ["males", "females"]
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, "sex", df_projections, df_diff_projections)
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "sex", df_projections, df_diff_projections)

    x = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
         '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, "age", df_projections, df_diff_projections)
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "age", df_projections, df_diff_projections)
    print("Canada", round(df_diff['avg age'].iloc[0], 2), "years in average age",
          df_stats_syn_pop_canada["avg age"].iloc[0], 'vs', df_stats_census_canada["avg age"].iloc[0])
    print("Canada", round(df_diff['med age'].iloc[0], 2), "years in med age",
          df_stats_syn_pop_canada["med age"].iloc[0], 'vs',
          df_stats_census_canada["med age"].iloc[0])

    x = ['< 20k $', '20k-60k $', '60k-100k $', '≥ 100k $']
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, "income")
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "income")

    x = ['1', '2', '3', '4', '5+']
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, "household size")
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "household size")
    print("Canada", round(df_diff['avg hh size'].iloc[0], 2), "person in average hh size",
          df_stats_syn_pop_canada["avg hh size"].iloc[0], 'vs', df_stats_census_canada["avg hh size"].iloc[0])

    x = ['Couples without children', 'Couples with children', 'One-parent-family', 'One-person', 'Other kind of hh']
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_diff, "household type")
    create_plot(x, df_stats_syn_pop_canada, df_stats_census_canada, df_errors, "household type")


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(ncol=2)


def compute_errors_cities(df_census, df_syn_pop):
    name = {'2016A00054611040': 'Winnipeg', '2016A00053520005': 'Toronto', '2016A00052443027': 'Sherbrooke'}
    stats_census = {}
    stats_syn_pop = {}
    diff_city = {}
    errors_city = {}

    cities = df_census.loc[df_census['area'].str.startswith('2016A0005'), 'area'].unique().tolist()
    cities.sort()

    for city in cities:
        print(city)
        print("DAs", sum(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals']) / len(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]),
              "% realistic on average",
              "from", round(min(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals']), 2), "to",
              round(max(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals']), 2),
              'q1', round(np.percentile(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals'], 25), 2), 'q3',
              round(np.percentile(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals'], 75), 2)
              , 'median', round(np.percentile(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals'], 50), 2), '5%',
              round(np.percentile(df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])]['% realistic individuals'], 5), 2))

        stats_syn_pop[city] = df_syn_pop.loc[df_syn_pop['area'].str.startswith("2016S0512" + city[9:13])].sum(axis=0)
        stats_syn_pop[city]['area'] = city
        stats_syn_pop[city] = stats_syn_pop[city].drop('% realistic individuals')

        stats_census[city] = df_census.loc[(df_census['area'] == str(city))]
        stats_syn_pop[city] = add_distribution_stats(pd.DataFrame(stats_syn_pop[city]).transpose())

        stats_census[city] = stats_census[city].set_index(['area'])
        stats_syn_pop[city] = stats_syn_pop[city].set_index(['area'])

        diff_city[city] = (stats_syn_pop[city].sub(stats_census[city]))
        errors_city[city] = diff_city[city].div(stats_census[city]) * 100.0
        errors_city[city] = errors_city[city].replace(np.inf, np.nan)
        diff_city[city] = diff_city[city].replace(np.inf, np.nan)

    for city in cities:
        x = ["Population", "Population\n private dwellings", "Households"]
        create_plot_count(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city])

        x = ["males", "females"]
        create_plot(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city] + " sex")

        x = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
             '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
        create_plot(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city] + " age")

        x = ['< 20k $', '20k-60k $', '60k-100k $', '≥ 100k $']
        create_plot(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city] + " income")

        x = ['1', '2', '3', '4', '5+']
        create_plot(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city] + " household size")

        x = ['Couples without children', 'Couples with children', 'One-parent-family', 'One-person', 'Other kind of hh']
        create_plot(x, stats_syn_pop[city], stats_census[city], errors_city[city], name[city] + " household type")


# Compute RMSE, re and pearson coff for each da of canada for each attrib
def compute_errors_DA(df_stats_census, df_stats_syn_pop):
    # Select only DAs and replace NaN stats by 0
    df_stats_census_da = df_stats_census.loc[df_stats_census['area'].str.startswith('2016S0512')]
    df_stats_census_da = df_stats_census_da.replace(np.inf, 0)
    df_stats_census_da = df_stats_census_da.replace(np.nan, 0)
    df_stats_syn_pop_da = df_stats_syn_pop.loc[df_stats_syn_pop['area'].str.startswith('2016S0512')]
    df_stats_syn_pop_da = df_stats_syn_pop_da.replace(np.inf, 0)
    df_stats_syn_pop_da = df_stats_syn_pop_da.replace(np.nan, 0)
    print("DAs", sum(df_stats_syn_pop_da['% realistic individuals']) / len(df_stats_syn_pop_da), "% realistic individuals on average",
          "from", round(min(df_stats_syn_pop_da['% realistic individuals']), 2), "to", round(max(df_stats_syn_pop_da['% realistic individuals']), 2),
          'q1', round(np.percentile(df_stats_syn_pop_da['% realistic individuals'],25), 2), 'q3', round(np.percentile(df_stats_syn_pop_da['% realistic individuals'],75), 2)
          , 'median', round(np.percentile(df_stats_syn_pop_da['% realistic individuals'],50), 2), '5%', round(np.percentile(df_stats_syn_pop_da['% realistic individuals'],5), 2))

    df_stats_syn_pop_da = df_stats_syn_pop_da.drop('% realistic individuals', axis=1)
    df_stats_census_da.sort_values(by=['area'], inplace=True)
    df_stats_syn_pop_da.sort_values(by=['area'], inplace=True)

    # Use area as index
    df_stats_census_da = df_stats_census_da.set_index(['area'])
    df_stats_syn_pop_da = df_stats_syn_pop_da.set_index(['area'])

    for column in df_stats_census_da:
        diff_da = []
        print(column)
        print(pearsonr(df_stats_census_da[column].values, df_stats_syn_pop_da[column].values))
        RMSE = mean_squared_error(df_stats_census_da[column].values, df_stats_syn_pop_da[column].values, squared=False)
        NRMSE = RMSE/(df_stats_census_da[column].max() - df_stats_census[column].min())
        print("Normalized Root Mean Square Error:", str(round(NRMSE*100, 4)))
        for index, row in df_stats_census_da.iterrows():
            if row[column] > 0:
                diff_da.append(abs(row[column] - df_stats_syn_pop_da.loc[index,column])/row[column]*100.0)

        print(np.mean(diff_da))
        print(round(np.min(diff_da),2), "/", round(np.percentile(diff_da, 25),2), "/", round(np.median(diff_da),2), "/",
              round(np.percentile(diff_da, 75),2), "/", round(np.max(diff_da),2))

    print(df_stats_census_da.corrwith(df_stats_syn_pop_da, axis=0))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    year = "2021"
    scenario = "LG" # need to change it in load_projections parameters as well
    print(year)

    # Load stats from census, synthetic population projections, estimates
    df_stats_census = load_stats_census(path, year)
    df_stats_syn_pop = load_stats_syn_pop(path, year, df_stats_census, scenario)
    df_projections = load_projections(path, 'LG: low-growth', year)
    df_estimates = load_estimates(path, year)

    # Check same DAs number
    nb_cities = 3
    print(len(df_stats_census.index)-nb_cities)
    print(len(df_stats_syn_pop.index))
    diff = pd.concat([df_stats_syn_pop, df_stats_census]).drop_duplicates(subset='area', keep=False)
    diff = diff.loc[diff['area'].str.startswith('2016S0512')]
    print(diff['area'])

    #df_stats_census.drop(diff.index, inplace=True)
    df_stats_syn_pop.drop(diff.index, inplace=True)

    # Compute errors DA level
    compute_errors_DA(df_stats_census, df_stats_syn_pop)

    # Compute errors Canada level
    compute_errors_Canada(df_stats_census, df_stats_syn_pop, df_projections, df_estimates)

    # Compute errors for each city
    compute_errors_cities(df_stats_census, df_stats_syn_pop)
