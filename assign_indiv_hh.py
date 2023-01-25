import random
import sys

import numpy as np
import pandas as pd
from pyreadstat import pyreadstat

# Load household microdata for all Canada
#https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/PYYXXR
def load_hh(path):
    dtafile = path + '/census_2016/PUMF/Census_2016_Hierarchial.dta'
    df_hh, meta = pyreadstat.read_dta(dtafile, usecols=["PP_ID", 'HH_ID', 'weight', 'agegrp', 'sex',
                                                        "hhsize", "prihm", "pr"])
    df_hh = df_hh.loc[df_hh["sex"] != 8]
    df_hh = df_hh.loc[df_hh["agegrp"] != 88]

    # add hhsize attribute to individuals by counting how many people are in their household
    df_hhsizes = df_hh.groupby(by=['HH_ID']).size() - 1
    df_hh['hhsize'] = df_hh['HH_ID'].map(df_hhsizes)

    return df_hh  # .loc[df_hh["pr"] == province]


# Load synthetic population for province
def load_syn_pop(path, year, filename, scenario):
    if year == "2016":
        file = path + '/' + filename + '/syn_pop/synthetic_pop_y_' + year + '.csv'
    else:
        file = path + '/' + filename + '/syn_pop/' + scenario + '/synthetic_pop_y_' + year + '.csv'
    df_pop = pd.read_csv(file)
    df_pop['area'] = df_pop['area'].astype(str)
    nb_age_grp = len(df_pop['agegrp'].unique()) - 1
    print(df_pop)
    # Add an HID column to all individuals
    df_pop.insert(0, 'HID', -1)
    # Add an agegrp_map column to all individuals
    df_pop = map_age(df_pop)

    # if live alone (hhsize 0) and not prihm (prihm 0), make the indiv prihm if adult or increase hhsize if child
    df_pop.loc[(df_pop['prihm'] == 0) & (df_pop['hhsize'] == 0) & (df_pop['agegrp'] > 2), 'prihm'] = 1
    df_pop.loc[(df_pop['prihm'] == 0) & (df_pop['hhsize'] == 0) & (df_pop['agegrp'] <= 2), 'hhsize'] = 1

    # Add an age column to all individuals
    for i in range(0, nb_age_grp):
        df_pop.loc[df_pop['agegrp'] == i, 'age'] = [random.randrange(i * 5, i * 5 + 5, 1) for k in
                                                    df_pop.loc[df_pop['agegrp'] == i].index]
    df_pop.loc[df_pop['agegrp'] == nb_age_grp, 'age'] = [np.random.geometric(p=0.2) + nb_age_grp * 5 - 1 for k in
                                                         df_pop.loc[df_pop['agegrp'] == nb_age_grp].index]

    df_pop.drop('Unnamed: 0', inplace=True, axis=1, errors="ignore")
    df_pop.drop('Unnamed: 0.1', inplace=True, axis=1, errors="ignore")
    # df_pop.insert(0, 'ID', range(0, len(df_pop)))
    return df_pop


# Load DA codes for province
def load_DAs(path):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    lookup['pr'] = lookup[' PRuid/PRidu'].astype(str)
    filtered_lookup = lookup.loc[lookup['pr'].str.strip() == province]
    place = filtered_lookup.iloc[0][" PRename/PRanom"]
    print(place)
    filename = place.replace(" ", "_").lower()
    DA_codes = filtered_lookup[' DAuid/ADidu'].astype(str).unique()
    DA_codes.sort()
    print(str(DA_codes.size) + " DAs")
    return DA_codes, filename


# Add an agegrp_map column with codes corresponding to the ones in households microdata
def map_age(df):
    df.loc[df["agegrp"].isin([0, 1]), "agegrp_map"] = 1
    df.loc[df["agegrp"].isin([11, 12]), "agegrp_map"] = 11
    df.loc[df["agegrp"].isin([13, 14]), "agegrp_map"] = 12
    df.loc[df["agegrp"].isin([15, 16, 17]), "agegrp_map"] = 13
    df.loc[(df["agegrp"] >= 2) & (df["agegrp"] <= 10), "agegrp_map"] = df["agegrp"]
    df["agegrp_map"] = df["agegrp_map"].astype(int)
    return df


def initialize_dict(agegrps, sexs, hhsizes):
    dict = {}
    for age in agegrps:
        dict[age] = {}
        for sex in sexs:
            sex = sex + 1
            dict[age][sex] = {}
            for hhsize in hhsizes:
                dict[age][sex][hhsize] = pd.DataFrame()
    return dict


def load_prihm_hh_probas(df_hh, df_indivs):
    # household responsible persons
    prihms = df_hh.loc[df_hh['prihm'] == 1]
    # non household responsible persons
    nonprihms = df_hh.loc[df_hh['prihm'] == 0]

    agegrps = df_indivs['agegrp_map'].unique()
    sexs = df_indivs['sex'].unique()
    hhsizes = df_indivs['hhsize'].unique()

    dict = initialize_dict(agegrps, sexs, hhsizes)

    # for each prihm age/sex/hhsize, compute probability of age/sex for other households members
    for age in agegrps:
        for sex in sexs:
            sex = sex + 1
            for hhsize in hhsizes:
                prihms_ = prihms.loc[(prihms['agegrp'] == age) & (prihms['sex'] == sex) & (prihms['hhsize'] == hhsize)]
                dict[age][sex][hhsize] = \
                    pd.DataFrame(
                        nonprihms.loc[nonprihms['HH_ID'].isin(prihms_['HH_ID'])].value_counts(["agegrp", "sex"],
                                                                                              normalize=True)).reset_index()
                dict[age][sex][hhsize].columns = ['agegrp', 'sex', 'proba']

    # To avoid convergence problems, allow zero states to be occupied with a very small probability
    for age in agegrps:
        for sex in sexs:
            sex = sex + 1
            for hhsize in hhsizes:
                for age_ in agegrps:
                    for sex_ in sexs:
                        sex_ = sex_ + 1
                        if (dict[age][sex][hhsize].loc[(age_ == dict[age][sex][hhsize]["agegrp"]) & (
                                sex_ == dict[age][sex][hhsize]["sex"])].empty):
                            data = {'agegrp': age_, 'sex': sex_, 'proba': 1.0 / len(df_hh.index)}
                            missing_row = pd.DataFrame([data])
                            dict[age][sex][hhsize] = pd.concat([dict[age][sex][hhsize], missing_row])
    return dict


def add_indiv(df_indivs, hh_subset, dict, prihm):
    sex_prihm = prihm['sex'] + 1
    hhsize_prihm = prihm['hhsize']
    age_grp_prihm = prihm['agegrp_map']

    if not hh_subset.empty:
        while True:
            rand = dict[age_grp_prihm][sex_prihm][hhsize_prihm].loc[
                (dict[age_grp_prihm][sex_prihm][hhsize_prihm]['agegrp'].isin(hh_subset['agegrp_map'].unique())) &
                (dict[age_grp_prihm][sex_prihm][hhsize_prihm]['sex'].isin(
                    set(map(lambda x: x + 1, hh_subset['sex']))))].sample(n=1, weights='proba')
            random_age = rand['agegrp'].values[0]
            random_sex = rand['sex'].values[0]
            ind = hh_subset.loc[(hh_subset['agegrp_map'] == random_age) & (hh_subset['sex'] + 1 == random_sex)]
            if not ind.empty:
                break

        df_indivs.at[ind.index[0], "HID"] = prihm['HID']
        hh_subset.at[ind.index[0], "HID"] = prihm['HID']
        df_indivs.at[ind.index[0], "hhsize"] = hhsize_prihm  # usefull for last hh assignments
        hh_subset = hh_subset.loc[hh_subset['HID'] == -1]
    return df_indivs, hh_subset


def complete_big_hh(df_indivs):
    for ind, row in df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['HID'] == -1) & (df_indivs['hhsize'] == 4)].iterrows():
        if not df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['prihm'] == 1) & (df_indivs['hhsize'] == 4)].empty:
            hh_random = df_indivs.loc[
                (df_indivs['area'] == code) & (df_indivs['prihm'] == 1) & (df_indivs['hhsize'] == 4)].sample(n=1)
            df_indivs.at[ind, "HID"] = hh_random['HID']
    return df_indivs


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    province = str(sys.argv[2])
    from_DA = int(sys.argv[3])
    year = sys.argv[4]
    scenario = sys.argv[5]

    DA_codes, filename = load_DAs(path)
    df_indivs = load_syn_pop(path, year, filename, scenario)
    df_hh = load_hh(path)
    dict = load_prihm_hh_probas(df_hh, df_indivs)

    hid = 0
    progress = from_DA + 1
    if from_DA == -1:
        from_DA = 0
        to_DA = len(DA_codes)
    else:
        to_DA = min(len(DA_codes), from_DA + 1000)

    df_indivs = df_indivs[df_indivs['area'].isin(DA_codes[from_DA:to_DA])]
    print(len(df_indivs.index))
    for code in DA_codes[from_DA:to_DA]:
        print(str(progress) + "/" + str(to_DA))
        progress = progress + 1
        print(code)

        # create households
        nb_hh = len(df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['prihm'] == 1)].index)
        df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['prihm'] == 1), "HID"] = range(hid, hid + nb_hh)
        prihms = df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['prihm'] == 1)]
        nonprihms = df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['prihm'] == 0)]
        hid = hid + nb_hh

        # Subsets of individual available to complete hh, by hhsize (and age)
        two_hh = nonprihms.loc[nonprihms['hhsize'] == 1]
        three_hh_child = nonprihms.loc[(nonprihms['hhsize'] == 2) & (nonprihms['agegrp_map'] < 4)]
        three_hh_adult = nonprihms.loc[(nonprihms['hhsize'] == 2) & (nonprihms['agegrp_map'] >= 4)]
        four_hh_child = nonprihms.loc[(nonprihms['hhsize'] == 3) & (nonprihms['agegrp_map'] < 4)]
        four_hh_adult = nonprihms.loc[(nonprihms['hhsize'] == 3) & (nonprihms['agegrp_map'] >= 4)]
        five_hh_child = nonprihms.loc[(nonprihms['hhsize'] == 4) & (nonprihms['agegrp_map'] < 4)]
        five_hh_adult = nonprihms.loc[(nonprihms['hhsize'] == 4) & (nonprihms['agegrp_map'] >= 4)]

        for id, prihm in prihms.iterrows():
            if prihm['hhsize'] == 1:
                df_indivs, two_hh = add_indiv(df_indivs, two_hh, dict, prihm)
            elif prihm['hhsize'] == 2:
                if not three_hh_child.empty:
                    df_indivs, three_hh_child = add_indiv(df_indivs, three_hh_child, dict, prihm)
                if not three_hh_adult.empty:
                    df_indivs, three_hh_adult = add_indiv(df_indivs, three_hh_adult, dict, prihm)
            elif prihm['hhsize'] == 3:
                if not four_hh_child.empty:
                    for i in range(0, 2):
                        df_indivs, four_hh_child = add_indiv(df_indivs, four_hh_child, dict, prihm)
                if not four_hh_adult.empty:
                    df_indivs, four_hh_adult = add_indiv(df_indivs, four_hh_adult, dict, prihm)
            elif prihm['hhsize'] == 4:
                if not five_hh_child.empty:
                    for i in range(0, 3):
                        df_indivs, five_hh_child = add_indiv(df_indivs, five_hh_child, dict, prihm)
                if not five_hh_adult.empty:
                    df_indivs, five_hh_adult = add_indiv(df_indivs, five_hh_adult, dict, prihm)

        # complete big households
        df_indivs = complete_big_hh(df_indivs)

        # complete prihm with remaining individuals
        indivs_without_hh = df_indivs.loc[(df_indivs['area'] == code) & (df_indivs['HID'] == -1)]
        for id, prihm in prihms.iterrows():
            while (prihm['hhsize'] > len(df_indivs.loc[df_indivs["HID"] == prihm['HID']]) - 1) & (
                    not indivs_without_hh.empty):
                df_indivs, indivs_without_hh = add_indiv(df_indivs, indivs_without_hh, dict, prihm)

    print("Indivs without hh (i.e. collective dwellings or not usual residents) " + str(
        len(indivs_without_hh)) + "/" + str(len(df_indivs.index)))
    df_indivs = df_indivs.drop(['agegrp_map'], axis=1)
    df_indivs = df_indivs[['HID','sex', 'prihm', 'agegrp', 'age','area', 'hdgree', 'lfact', 'hhsize', 'totinc']]

    if year == "2016":
        output_path = path + '/' + filename + '/syn_pop'
    else:
        output_path = path + "/" + filename + '/syn_pop/' + scenario

    if not df_indivs.empty:
        if (from_DA == 0) & (to_DA == len(DA_codes)):
            df_indivs.to_csv(output_path + "/synthetic_pop_" + str(year) + "_hh.csv", index=False)
        else:
            df_indivs.to_csv(output_path + "/synthetic_pop_" + str(year) + "_" + str(to_DA) + "_hh.csv", index=False)