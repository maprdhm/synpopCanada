import math
import os
import random
import sys
import time
from distutils.util import strtobool

import humanleague
import numpy as np
import pandas as pd
import pyreadstat

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
total_vb_id = total_age_by_sex_vb_id = total_hh_vb_id = age_vb = hdgree_vb = lfact_vb = hhsize_vb = totinc_vb = cfstat_vb = ""

regions = {'10': 'ATLANTIC', '11': 'ATLANTIC', '12': 'ATLANTIC', '13': 'ATLANTIC',
           '24': 'QUEBEC',
           '35': 'ONTARIO',
           '46': 'PRAIRIES', '47': 'PRAIRIES', '48': 'PRAIRIES',
           '59': 'BRITISH_COLUMBIA',
           '60': 'TERRITORIES', '61': 'TERRITORIES', '62': 'TERRITORIES'}


# Load individuals microdata for province
# Download from https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/GDJRT8
def load_indiv(path):
    dtafile = path + '/census_2016/PUMF/Census_2016_Individual_PUMF.dta'
    df_indiv, meta_indiv = pyreadstat.read_dta(dtafile, usecols=['ppsort', 'weight', 'agegrp', 'Sex',
                                                                 "hdgree", "lfact", 'TotInc',
                                                                 "hhsize", "cfstat", "prihm",
                                                                 "cma", "pr"])
    df_indiv['pr'] = df_indiv['pr'].astype(str)
    if (province == '60') | (province == '61') | (province == '62'):
        df_indiv = df_indiv.loc[df_indiv["pr"].str.strip() == '70']
    else:
        df_indiv = df_indiv.loc[df_indiv["pr"].str.strip() == province]
    return df_indiv


# Load DA codes for province
# Download from https://www12.statcan.gc.ca/census-recensement/alternative_alternatif.cfm?l=eng&dispext=zip&teng=2016_92-151_XBB_csv.zip&loc=http://www12.statcan.gc.ca/census-recensement/2016/geo/ref/gaf/files-fichiers/2016_92-151_XBB_csv.zip&k=%20%20%20%2023271
def load_DAs(path):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    lookup['pr'] = lookup[' PRuid/PRidu'].astype(str)
    filtered_lookup = lookup.loc[lookup['pr'].str.strip() == province]
    place = filtered_lookup.iloc[0][" PRename/PRanom"]
    print(place)
    filename = place.replace(" ", "_").lower()
    DA_codes = filtered_lookup[' DAuid/ADidu'].unique()
    DA_codes.sort()
    print(str(DA_codes.size) + " DAs")
    return DA_codes, filename


# this is a copy-paste from household_microsynth
def unlistify(table, columns, sizes, values):
    """
    Converts an n-column table of counts into an n-dimensional array of counts
    """
    pivot = table.pivot_table(index=columns, values=values, aggfunc='sum')
    # order must be same as column order above
    array = np.zeros(sizes, dtype=int)
    array[tuple(pivot.index.codes)] = pivot.values.flat
    return array


# Map ages to 18 classes
def map_age_grp(df_indiv):
    for i in range(17, 22):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 8
    for i in range(16, 7, -1):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 7
    df_indiv.loc[df_indiv["agegrp"] == 1, "agegrp"] = 10
    df_indiv.loc[df_indiv["agegrp"] == 2, "agegrp"] = 11
    df_indiv.loc[df_indiv["agegrp"] == 3, "agegrp"] = 11
    df_indiv.loc[df_indiv["agegrp"] == 4, "agegrp"] = 12
    df_indiv.loc[df_indiv["agegrp"] == 5, "agegrp"] = 12
    df_indiv.loc[df_indiv["agegrp"] == 6, "agegrp"] = 14
    df_indiv.loc[df_indiv["agegrp"] == 7, "agegrp"] = 14
    df_indiv = df_indiv.loc[df_indiv["agegrp"] != 88]
    return df_indiv


# Map ages to 7 classes
def map_age_grp_new(df_indiv):
    for i in range(17, 22):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 8

    for i in range(16, 7, -1):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = 13

    df_indiv.loc[df_indiv["agegrp"] == 1, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 2, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 3, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 4, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 5, "agegrp"] = 9

    df_indiv.loc[df_indiv["agegrp"] == 6, "agegrp"] = 13
    df_indiv.loc[df_indiv["agegrp"] == 7, "agegrp"] = 13
    df_indiv = df_indiv.loc[df_indiv["agegrp"] != 88]
    return df_indiv


# Map hdgree to 4 classes
def map_hdgree(df_indiv):
    df_indiv.loc[df_indiv["hdgree"] == 88, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] == 99, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] > 2, "hdgree"] = 1686
    df_indiv.loc[df_indiv["hdgree"] == 1, "hdgree"] = 1684
    df_indiv.loc[df_indiv["hdgree"] == 2, "hdgree"] = 1685
    return df_indiv


# Map lfact to 3 classes
def map_lfact(df_indiv):
    df_indiv.loc[df_indiv["lfact"] == 1, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] == 2, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] < 11, "lfact"] = 1868
    df_indiv.loc[df_indiv["lfact"] < 100, "lfact"] = 1869
    return df_indiv


# Map hhsize to 5 classes
def map_hhsize(df_indiv):
    df_indiv.loc[df_indiv["hhsize"] == 8, "hhsize"] = 1
    df_indiv.loc[df_indiv["hhsize"] > 5, "hhsize"] = 5
    return df_indiv


# Map totinc to 4 classes
def map_totinc(df_indiv):
    df_indiv = df_indiv.loc[df_indiv["TotInc"] != 88888888]
    df_indiv.loc[df_indiv["TotInc"] == 99999999, "TotInc"] = 695
    df_indiv.loc[df_indiv["TotInc"] < 20000, "TotInc"] = 695

    # for i in range(1, 10):
    #    df_indiv.loc[((df_indiv["TotInc"] >= 10000 * i) & (df_indiv["TotInc"] < 10000 * (i + 1))), "TotInc"] = 695 + i

    df_indiv.loc[((df_indiv["TotInc"] >= 20000) & (df_indiv["TotInc"] < 60000)), "TotInc"] = 697
    df_indiv.loc[((df_indiv["TotInc"] >= 60000) & (df_indiv["TotInc"] < 100000)), "TotInc"] = 701
    df_indiv.loc[df_indiv["TotInc"] >= 100000, "TotInc"] = 705

    return df_indiv


def map_cfstat(df_indiv):
    df_indiv.loc[df_indiv["cfstat"] == 8, "cfstat"] = 7
    return df_indiv


# Load seed from microsample
def load_seed(df_indiv, fast):
    df_indiv = map_age_grp(df_indiv)
    # df_indiv = map_age_grp_new(df_indiv)

    df_indiv = map_hdgree(df_indiv)
    df_indiv = map_lfact(df_indiv)
    df_indiv = map_hhsize(df_indiv)
    df_indiv = map_totinc(df_indiv)
    df_indiv = map_cfstat(df_indiv)

    n_sex = len(df_indiv['Sex'].unique())
    n_age = len(df_indiv['agegrp'].unique())
    n_prihm = len(df_indiv['prihm'].unique())
    n_hdgree = len(df_indiv['hdgree'].unique())
    n_lfact = len(df_indiv['lfact'].unique())
    n_hhsize = len(df_indiv['hhsize'].unique())
    n_totinc = len(df_indiv['TotInc'].unique())
    n_cfstat = len(df_indiv['cfstat'].unique())

    cols = ["Sex", "prihm", 'agegrp', "hdgree", "lfact", "hhsize", "TotInc"]
    shape = [n_sex, n_prihm, n_age, n_hdgree, n_lfact, n_hhsize, n_totinc]
    if fast:
        cols = ["Sex", "prihm", 'agegrp', "hdgree", "lfact", "hhsize", "TotInc", "cfstat"]
        shape = [n_sex, n_prihm, n_age, n_hdgree, n_lfact, n_hhsize, n_totinc, n_cfstat]

    seed = unlistify(df_indiv, cols, shape, "weight")

    # Convergence problems can occur when one of the rows is zero yet the marginal total is nonzero.
    # Can get round this by adding a small number to the seed effectively allowing zero states
    #  to be occupied with a finite probability
    seed = seed.astype(float) + 1.0  # / np.sum(seed)
    if fast:
        seed = seed * get_impossible(seed)

    return seed


def get_impossible(seed):
    # zeros out impossible states, all others are equally probable
    constraints = np.ones(seed.shape)
    # Add impossible constraints:
    # prihm 1 and age 0 to 2,
    constraints[:, 1, 0, :, :, :, :, :] = 0
    constraints[:, 1, 1, :, :, :, :, :] = 0
    constraints[:, 1, 2, :, :, :, :, :] = 0
    # hdgree >0 and age 0 to 2,
    constraints[:, :, 0, 1, :, :, :, :] = 0
    constraints[:, :, 0, 2, :, :, :, :] = 0
    constraints[:, :, 1, 1, :, :, :, :] = 0
    constraints[:, :, 1, 2, :, :, :, :] = 0
    constraints[:, :, 2, 1, :, :, :, :] = 0
    constraints[:, :, 2, 2, :, :, :, :] = 0
    # employed or unemployed and age 0 to 2
    constraints[:, :, 0, :, 0, :, :, :] = 0
    constraints[:, :, 0, :, 1, :, :, :] = 0
    constraints[:, :, 1, :, 0, :, :, :] = 0
    constraints[:, :, 1, :, 1, :, :, :] = 0
    constraints[:, :, 2, :, 0, :, :, :] = 0
    constraints[:, :, 2, :, 1, :, :, :] = 0
    # hhsize 0 (1p) and age 0 to 2
    constraints[:, :, 0, :, :, 0, :, :] = 0
    constraints[:, :, 1, :, :, 0, :, :] = 0
    constraints[:, :, 2, :, :, 0, :, :] = 0
    # prihm 0 and hhsize 0 (1p)
    constraints[:, 0, :, :, :, 0, :, :] = 0
    # totinc >0 and age 0 to 2
    for i in range(1, 4):
        constraints[:, :, 0, :, :, :, i, :] = 0
        constraints[:, :, 1, :, :, :, i, :] = 0
        constraints[:, :, 2, :, :, :, i, :] = 0
    # cfstat 5 (1p) hhsize >0 (1p)
    for i in range(1, 5):
        constraints[:, :, :, :, :, i, :, 5] = 0

    return constraints


# Load census 2016 profile for the province
def load_census_profile(path, region):
    start_rows = pd.read_csv(
        path + '/census_2016/98-401-X2016044_' + region+'_eng_CSV/Geo_starting_row_' + region+'_CSV.csv',
        dtype=str)
    start = int(start_rows.loc[start_rows['Geo Code'] == province]['Line Number'].values[0])
    end = int(start_rows.loc[start_rows['Geo Code'].str.startswith(province)]['Line Number'].values[-1])
    census = pd.read_csv(
        path + '/census_2016/98-401-X2016044_' + region+'_eng_CSV/98-401-X2016044_'+ region+'_English_CSV_data.csv',
        skiprows=range(1,start-1), nrows=end - start, low_memory=False,
        usecols=["GEO_CODE (POR)",
                 "DIM: Profile of Dissemination Areas (2247)",
                 "Member ID: Profile of Dissemination Areas (2247)",
                 "Dim: Sex (3): Member ID: [1]: Total - Sex",
                 "Dim: Sex (3): Member ID: [2]: Male",
                 "Dim: Sex (3): Member ID: [3]: Female"
                 ]
    )
    census.rename(columns={'GEO_CODE (POR)': 'geocode',
                           'DIM: Profile of Dissemination Areas (2247)': 'variable',
                           'Member ID: Profile of Dissemination Areas (2247)': 'variableId',
                           'Dim: Sex (3): Member ID: [1]: Total - Sex': 'total',
                           'Dim: Sex (3): Member ID: [2]: Male': 'totalMale',
                           'Dim: Sex (3): Member ID: [3]: Female': 'totalFemale'}, inplace=True)
    return census


# Load variables identifiants in census
def load_vbs_ids(census):
    # Total population id
    total_vb_id = census.loc[census["variable"] == "Population, 2016"]['variableId'].iloc[0]

    # Total age by sex id
    total_ageby_sex_vb_id = census.loc[
        census["variable"] == "Total - Age groups and average age of the population - 100% data"]['variableId'].iloc[0]

    # Total households id
    total_hh_vb_id = \
        census.loc[census["variable"] == "Private dwellings occupied by usual residents"]['variableId'].iloc[0]

    # Total by age id
    age_vb = {}
    for i in range(0, 85, 5):
        age_vb[i] = census.loc[census["variable"] ==
                               str(i) + " to " + str(i + 4) + " years"]['variableId'].iloc[0]
    age_vb[85] = census.loc[census["variable"] == "85 years and over"]['variableId'].iloc[0]
    '''
    age_vb = {}
    age_vb[0] = 9
    age_vb[1] = 13
    age_vb[2] = 25
    age_vb[3] = 26
    age_vb[4] = 27
    age_vb[5] = 28
    age_vb[6] = 29
    '''

    # Total by hdgree id
    hdgree_vb = {}
    id_start = census.loc[census["variable"] == "Total - Highest certificate, diploma or degree for the population " \
                                                "aged 15 years and over in private households - 25% sample data"][
                   'variableId'].iloc[0] + 1
    for i in range(0, 3):
        hdgree_vb[i] = id_start + i

    # Total by lfact id
    lfact_vb = {}
    id_start = census.loc[census["variable"] == "Total - Population aged 15 years and over by Labour force status - " \
                                                "25% sample data"]['variableId'].iloc[0] + 2
    for i in range(0, 3):
        lfact_vb[i] = id_start + i

    # Total by hhsize id
    hhsize_vb = {}
    id_start = \
        census.loc[census["variable"] == "Total - Private households by household size - 100% data"]['variableId'].iloc[
            0] + 1
    for i in range(0, 5):
        hhsize_vb[i] = id_start + i

    # Total by totinc id
    totinc_vb = {}
    id_start = census.loc[census["variable"] ==
                          "Total - Total income groups in 2015 for the population aged 15 years and over in private " \
                          "households - 100% data"]['variableId'].iloc[0] + 4
    # for i in range(0, 6):
    #    totinc_vb[i] = id_start + i*2
    totinc_vb[0] = id_start
    totinc_vb[1] = id_start + 2
    totinc_vb[2] = id_start + 6
    totinc_vb[3] = id_start + 10

    # Total by cfstat id
    cfstatVb = {}
    cfstatVb[0] = census.loc[census["variable"] == "Couples without children"]['variableId'].iloc[0]
    cfstatVb[1] = census.loc[census["variable"] == "Couples with children"]['variableId'].iloc[0]
    cfstatVb[2] = census.loc[census["variable"] == "Total - Lone-parent census families in private households - 100% " \
                                                   "data"]['variableId'].iloc[0]
    cfstatVb[3] = cfstatVb[1] + 1  # +2+3
    cfstatVb[4] = cfstatVb[2] + 1  # +2+3
    cfstatVb[5] = census.loc[census["variable"] == "One-person households"]['variableId'].iloc[0]
    cfstatVb[6] = census.loc[census["variable"] == "Two-or-more person non-census-family households"][
        'variableId'].iloc[0]

    return total_vb_id, total_ageby_sex_vb_id, total_hh_vb_id, age_vb, hdgree_vb, lfact_vb, hhsize_vb, totinc_vb, cfstatVb


def load_province_marginals(da_census, province_census):
    total_age = {}
    total_age_f = {}
    total_age_m = {}
    total_hh_size = {}

    population_province = int(province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0])

    total_pop = int(da_census.loc[da_census["variableId"] == total_vb_id]['total'].iloc[0])
    total_male = int(total_pop * int(int(
        province_census.loc[province_census["variableId"] == total_age_by_sex_vb_id]['totalMale'].iloc[
            0]) / population_province))
    total_female = total_pop - total_male

    for i in range(0, 86, 5):
        total_age[i] = int(total_pop * int(
            province_census.loc[province_census["variableId"] == age_vb[i]]['total'].iloc[0]) / population_province)
        total_age_m[i] = int(total_age[i] * int(
            province_census.loc[province_census["variableId"] == age_vb[i]]['totalMale'].iloc[0]) / population_province)
        total_age_f[i] = total_age[i] - total_age_m[i]
    '''
    for i in range(0, len(age_vb)):
        total_age[i] = int(total_pop * int(
            province_census.loc[province_census["variableId"] == age_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
        total_age_m[i] = int(total_age[i] * int(
            province_census.loc[province_census["variableId"] == age_vb[i]]['totalMale'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == age_vb[i]]['total'].iloc[0]))
        total_age_f[i] = total_age[i] - total_age_m[i]
    '''

    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = int(total_pop * int(int(province_census.loc[province_census["variableId"] == hhsize_vb[i]][
                                                       'total'].iloc[0]) * (i + 1)) / population_province)

    return total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size


def load_da_marginals(da_census):
    total_age = {}
    total_age_f = {}
    total_age_m = {}
    total_hh_size = {}

    total_pop = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['total'].iloc[0])
    print(str(total_pop) + " individuals in the DA")
    total_male = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['totalMale'].iloc[0])
    total_female = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['totalFemale'].iloc[0])

    for i in range(0, 86, 5):
        # for i in range(0, len(age_vb)):
        total_age[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['total'].iloc[0])
        total_age_m[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['totalMale'].iloc[0])
        total_age_f[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['totalFemale'].iloc[0])

    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = int(da_census.loc[da_census["variableId"] == hhsize_vb[i]]['total'].iloc[0]) * (i + 1)
    return total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size


def load_marginals_age_sex_hh(da_census, province_census):
    # if data for DA not available, use distribution of province
    total_pop_value = da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['total'].iloc[0]
    if (total_pop_value == "x") or (total_pop_value == "F"):
        print("Census data not available for DA population, use province data")
        return load_province_marginals(da_census, province_census)
    else:
        return load_da_marginals(da_census)


def load_marginals_hdegree(da_census, province_census, total_pop):
    total_hdgree = {}
    # if data for DA not available, use distribution of province
    total_hdegree_value = da_census.loc[da_census["variableId"] == hdgree_vb[0]]['total'].iloc[0]
    if (total_hdegree_value == "x") or (total_hdegree_value == "F"):
        print("Census data not available for DA higher degree, use province data")
        for i in range(0, len(hdgree_vb)):
            total_hdgree[i] = int(total_pop * int(
                province_census.loc[province_census["variableId"] == hdgree_vb[i]]['total'].iloc[0]) / int(
                province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
    else:
        for i in range(0, len(hdgree_vb)):
            total_hdgree[i] = int(da_census.loc[da_census["variableId"] == hdgree_vb[i]]['total'].iloc[0])
    return total_hdgree


def load_marginals_lfact(da_census, province_census, total_pop):
    total_lfact = {}
    # if data for DA not available, use distribution of province
    total_lfact_value = da_census.loc[da_census["variableId"] == lfact_vb[0]]['total'].iloc[0]
    if (total_lfact_value == "x") or (total_lfact_value == "F"):
        print("Census data not available for DA employment, use province data")
        for i in range(0, len(lfact_vb)):
            total_lfact[i] = int(total_pop * int(
                province_census.loc[province_census["variableId"] == lfact_vb[i]]['total'].iloc[0]) / int(
                province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
    else:
        for i in range(0, len(lfact_vb)):
            total_lfact[i] = int(da_census.loc[da_census["variableId"] == lfact_vb[i]]['total'].iloc[0])
    return total_lfact


def load_marginals_totinc(da_census, province_census, total_pop):
    total_inc = {}
    # if data for DA not available, use distribution of province
    total_totinc_value = da_census.loc[da_census["variableId"] == totinc_vb[0]]['total'].iloc[0]
    if (total_totinc_value == "x") or (total_totinc_value == "F"):
        print("Census data not available for DA income, use province data")
        total_pop_prov = int(province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0])
        # for i in range(0, len(totinc_vb)):
        # total_inc[i] = int(total_pop * int((
        #    province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0])) / int(
        #    province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
        total_inc[0] = int(total_pop * int(
            int(province_census.loc[province_census["variableId"] == totinc_vb[0]]['total'].iloc[0]) +
            int(province_census.loc[province_census["variableId"] == (totinc_vb[0] + 1)]['total'].iloc[
                    0])) / total_pop_prov)
        for i in range(1, 3):
            total_inc[i] = int(total_pop * int(
                int(province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 1)]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 2)]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 3)]['total'].iloc[
                        0])) / total_pop_prov)
        total_inc[3] = int(total_pop * int(
            int(province_census.loc[province_census["variableId"] == totinc_vb[3]]['total'].iloc[0])) / total_pop_prov)
    else:
        # for i in range(0, len(totinc_vb)):
        # total_inc[i] = int(da_census.loc[da_census["variableId"] == totinc_vb[i]]['total'].iloc[0])
        total_inc[0] = int(da_census.loc[da_census["variableId"] == totinc_vb[0]]['total'].iloc[0]) + \
                       int(da_census.loc[da_census["variableId"] == (totinc_vb[0] + 1)]['total'].iloc[0])
        for i in range(1, 3):
            total_inc[i] = int(da_census.loc[da_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 1)]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 2)]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 3)]['total'].iloc[0])
        total_inc[3] = int(da_census.loc[da_census["variableId"] == totinc_vb[3]]['total'].iloc[0])
    return total_inc


def load_marginals_cfstat(da_census, province_census, total_pop):
    total_cfstat = {}
    total_cfstat_value = da_census.loc[da_census["variableId"] == cfstat_vb[0]]['total'].iloc[0]
    # if data for DA not available, use distribution of province
    if (total_cfstat_value == "x") or (total_cfstat_value == "F"):
        print("Census data not available for DA census family, use province data")
        total_pop_prov = int(province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0])
        for i in range(0, 2):
            total_cfstat[i] = int(total_pop * int(
                (province_census.loc[province_census["variableId"] == cfstat_vb[i]]['total'].iloc[
                    0]) * 2) / total_pop_prov)
        total_cfstat[2] = int(total_pop * int(
            (province_census.loc[province_census["variableId"] == cfstat_vb[2]]['total'].iloc[0])) / total_pop_prov)
        for i in range(3, 5):
            total_cfstat[i] = int(total_pop * int(
                int(province_census.loc[province_census["variableId"] == cfstat_vb[i]]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == cfstat_vb[i] + 1]['total'].iloc[0]) * 2 +
                int(province_census.loc[province_census["variableId"] == cfstat_vb[i] + 2]['total'].iloc[0]) * 3
            ) / total_pop_prov)
        total_cfstat[5] = int(total_pop * int(
            (province_census.loc[province_census["variableId"] == cfstat_vb[5]]['total'].iloc[0])) / int(
            province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
        total_cfstat[6] = int(total_pop * int(
            (province_census.loc[province_census["variableId"] == cfstat_vb[6]]['total'].iloc[0]) * 2) / total_pop_prov)
    else:
        for i in range(0, 2):
            total_cfstat[i] = int(da_census.loc[da_census["variableId"] == cfstat_vb[i]]['total'].iloc[0]) * 2
        total_cfstat[2] = int(da_census.loc[da_census["variableId"] == cfstat_vb[2]]['total'].iloc[0])
        for i in range(3, 5):
            total_cfstat[i] = int(da_census.loc[da_census["variableId"] == cfstat_vb[i]]['total'].iloc[0]) + \
                              int(da_census.loc[da_census["variableId"] == (cfstat_vb[i] + 1)]['total'].iloc[0]) * 2 + \
                              int(da_census.loc[da_census["variableId"] == (cfstat_vb[i] + 2)]['total'].iloc[0]) * 3
        total_cfstat[5] = int(da_census.loc[da_census["variableId"] == cfstat_vb[5]]['total'].iloc[0])
        total_cfstat[6] = int(da_census.loc[da_census["variableId"] == cfstat_vb[6]]['total'].iloc[0]) * 2
    return total_cfstat


def add_missing_hdegree(total_hdgree, province_census, total_age, total_pop):
    # add no diploma count for < 15y
    total_hdgree[0] += total_age[0] + total_age[5] + total_age[10]
    # total_hdgree[0] += total_age[0] # to use if 7 age classes

    # add missing hdegree according to province distribution
    miss = total_pop - sum(total_hdgree.values())
    for i in range(0, len(hdgree_vb)):
        total_hdgree[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == hdgree_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == hdgree_vb[0] - 1]['total'].iloc[0]))
        if total_hdgree[i] < 0:
            total_hdgree[i] = 0
    while total_pop != sum(total_hdgree.values()):
        miss = total_pop - sum(total_hdgree.values())
        random_key = random.sample(list(total_hdgree), 1)[0]
        if miss > 0 or total_hdgree[random_key] > 0:
            total_hdgree[random_key] += math.copysign(1, miss)
    return total_hdgree


def add_missing_lfact(total_lfact, province_census, total_age, total_pop):
    # add no labour force count for < 15y
    total_lfact[0] += total_age[0] + total_age[5] + total_age[10]
    # total_lfact[0] += total_age[0] # to use if 7 age classes

    # add missing labour force status following province distribution
    miss = total_pop - sum(total_lfact.values())
    for i in range(0, len(lfact_vb)):
        total_lfact[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == lfact_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == lfact_vb[0] - 2]['total'].iloc[0]))
        if total_lfact[i] < 0:
            total_lfact[i] = 0
    while total_pop != sum(total_lfact.values()):
        miss = total_pop - sum(total_lfact.values())
        random_key = random.sample(list(total_lfact), 1)[0]
        if miss > 0 or total_lfact[random_key] > 0:
            total_lfact[random_key] += math.copysign(1, miss)
    return total_lfact


def add_missing_hhsize(total_hh_size, total_pop):
    # add missing hhsize in 5+ class
    miss = total_pop - sum(total_hh_size.values())
    if (miss > 0):
        total_hh_size[len(hhsize_vb) - 1] += int(miss)

    while total_pop != sum(total_hh_size.values()):
        miss = total_pop - sum(total_hh_size.values())
        random_key = random.sample(list(total_hh_size), 1)[0]
        if (miss > 0 and random_key >= miss - 1):
            random_key = miss
        if miss > 0 or total_hh_size[random_key] > random_key + 1:
            total_hh_size[random_key] += math.copysign(random_key + 1, miss)
    return total_hh_size


def add_missing_totinc(total_inc, province_census, total_age, total_pop):
    # add <20k income count for < 15y
    total_inc[0] += total_age[0] + total_age[5] + total_age[10]
    # total_inc[0] += total_age[0] # to use if 7 age classes

    # add missing income following province distribution
    miss = total_pop - sum(total_inc.values())
    for i in range(0, len(totinc_vb)):
        total_inc[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == totinc_vb[0] - 2]['total'].iloc[0]))
        if total_inc[i] < 0:
            total_inc[i] = 0
    while total_pop != sum(total_inc.values()):
        miss = total_pop - sum(total_inc.values())
        random_key = random.sample(list(total_inc), 1)[0]
        if miss > 0 or total_inc[random_key] > 0:
            total_inc[random_key] += math.copysign(1, miss)
    return total_inc


def add_missing_cfstat(total_cfstat, total_pop):
    # add missing cfstat in last class
    miss = total_pop - sum(total_cfstat.values())
    if (miss > 0):
        total_cfstat[len(cfstat_vb) - 1] += int(miss)
    while total_pop != sum(total_cfstat.values()):
        miss = total_pop - sum(total_cfstat.values())
        random_key = random.sample(list(total_cfstat), 1)[0]
        if miss > 0 or total_cfstat[random_key] > 0:
            total_cfstat[random_key] += math.copysign(1, miss)
    return total_cfstat


# Increment or decrement male/female count to match total
def match_sex_count_total(total_pop, total_male, total_female):
    if total_pop != total_male + total_female:
        miss = total_pop - total_male - total_female
        total_male += int(miss / 2)
        total_female = total_pop - total_male
    return total_male, total_female


# Increment or decrement age counts to match total
def match_age_count_total(total_pop, total_male, total_female, total_age, total_age_m, total_age_f):
    for i in range(0, 86, 5):
        # for i in range(0, len(age_vb)):
        total_age[i] = total_age_m[i] + total_age_f[i]
    while total_pop != sum(total_age.values()):
        miss = total_pop - sum(total_age.values())
        random_key = random.sample(list(total_age), 1)[0]
        if miss > 0 or total_age[random_key] > 0:
            if (total_male < sum(total_age_m.values()) and miss < 0) or (
                    total_male > sum(total_age_m.values()) and miss > 0):
                if miss > 0 or total_age_m[random_key] > 0:
                    total_age_m[random_key] = total_age_m[random_key] + math.copysign(1, miss)
            elif (total_female < sum(total_age_f.values()) and miss < 0) or (
                    total_female > sum(total_age_f.values()) and miss > 0):
                if miss > 0 or total_age_f[random_key] > 0:
                    total_age_f[random_key] = total_age_f[random_key] + math.copysign(1, miss)
            for i in range(0, 86, 5):
                # for i in range(0, len(age_vb)):
                total_age[i] = total_age_m[i] + total_age_f[i]
    total_male = sum(total_age_m.values())
    total_female = sum(total_age_f.values())
    return total_age, total_age_m, total_age_f, total_male, total_female


# NOT USED
# Find best rounding threshold
# Issue: stuck in local minima
def comb_opti_integerization(p, total_pop):
    # increase threshold while error decreases
    threshold = 0.1
    previous_err = total_pop
    p["result_"] = np.around(p["result"] - threshold + 0.5)
    a = humanleague.flatten(p["result_"])[0]
    err = (abs(total_pop - len(a)))
    while (err < previous_err) | (err / total_pop * 100 > 10):
        threshold = round(threshold + 0.05, 2)
        previous_err = err
        p["result_"] = np.around(p["result"] - threshold + 0.5)
        a = humanleague.flatten(p["result_"])[0]
        err = (abs(total_pop - len(a)))

    # decrease threshold by smaller steps while error decreases
    threshold = round(threshold - 0.01, 2)
    previous_err = err
    p["result_"] = np.around(p["result"] - threshold + 0.5)
    a = humanleague.flatten(p["result_"])[0]
    err = (abs(total_pop - len(a)))
    while err < previous_err:
        threshold = round(threshold - 0.01, 2)
        previous_err = err
        p["result_"] = np.around(p["result"] - threshold + 0.5)
        a = humanleague.flatten(p["result_"])[0]
        err = (abs(total_pop - len(a)))

    threshold = round(threshold + 0.01, 2)
    print(str(previous_err / total_pop * 100) + " % error in individuals count")
    p["result"] = np.around(p["result"] - threshold + 0.5)
    return p["result"]


def probabilistic_sampling(p, total_pop):
    probas = np.float64(p["result"]).ravel()
    probas /= np.sum(probas)
    selected = np.random.choice(len(probas), total_pop, False, probas)
    result = np.zeros(p["result"].shape, np.uint8)
    result.ravel()[selected] = 1
    return result


def synthetise_pop_da(syn_inds, DA_code, da_census, province_census, seed, fast):
    total_hh = int(da_census.loc[da_census["variableId"] == total_hh_vb_id]['total'].iloc[0])
    total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size = load_marginals_age_sex_hh(
        da_census, province_census)
    total_hdgree = load_marginals_hdegree(da_census, province_census, total_pop)
    total_lfact = load_marginals_lfact(da_census, province_census, total_pop)
    total_inc = load_marginals_totinc(da_census, province_census, total_pop)
    total_cfstat = load_marginals_cfstat(da_census, province_census, total_pop)

    total_hh = min(total_pop, total_hh)
    print("Add missing hdegree...")
    total_hdgree = add_missing_hdegree(total_hdgree, province_census, total_age, total_pop)
    print("Add missing lfact...")
    total_lfact = add_missing_lfact(total_lfact, province_census, total_age, total_pop)
    print("Add missing hhsize...")
    total_hh_size = add_missing_hhsize(total_hh_size, total_pop)
    print("Add missing income...")
    total_inc = add_missing_totinc(total_inc, province_census, total_age, total_pop)
    print("Add missing cfstat...")
    total_cfstat = add_missing_cfstat(total_cfstat, total_pop)

    print("Match sex counts to total...")
    total_male, total_female = match_sex_count_total(total_pop, total_male, total_female)
    print("Match age counts to total...")
    total_age, total_age_m, total_age_f, total_male, total_female = match_age_count_total(total_pop, total_male,
                                                                                          total_female, total_age,
                                                                                          total_age_m, total_age_f)

    print("Gather marginals...")
    # get marginal by sex, by prihm, by age, by agebysex, hdgree
    # 0:F 1:M
    marginal_sex = np.array([total_female, total_male])
    # 0:no 1:yes
    marginal_prihm = np.array([total_pop - total_hh, total_hh])
    # 0:0-4y ... 17: 85+
    marginal_age = np.array(list(total_age.values()))
    # 0: F age, 1: M age
    marginal_age_by_sex = np.array([list(total_age_f.values()), list(total_age_m.values())])
    # 0: no, 1:secondary, 2: university
    marginal_hdgree = np.array(list(total_hdgree.values()))
    # 0: employed, 1:unemployed, 2: not in labour force
    marginal_lfact = np.array(list(total_lfact.values()))
    # 0: 1; 1: 2; 2: 3; 3: 4; 4: 5+
    marginal_hh_size = np.array(list(total_hh_size.values()))
    # <20k, 20-60k, 60-100k, 100+
    marginal_inc = np.array(list(total_inc.values()))
    # 0 partner no child, 1 partner with child, 2 lone parent, 3 child of couple, 4 child of lone, 5 alone, 6 other
    marginal_cfstat = np.array(list(total_cfstat.values()))

    i0 = np.array([0])
    i1 = np.array([1])
    i2 = np.array([2])
    i3 = np.array([0, 2])
    i4 = np.array([3])
    i5 = np.array([4])
    i6 = np.array([5])
    i7 = np.array([6])
    i8 = np.array([7])

    if fast:
        print("Apply IPF (could be replaced by qisi for more accurate.)")
        p = humanleague.ipf(seed, [i0, i1, i2, i3, i4, i5, i6, i7, i8],
                            [marginal_sex.astype(float), marginal_prihm.astype(float), marginal_age.astype(float),
                             marginal_age_by_sex.astype(float), marginal_hdgree.astype(float),
                             marginal_lfact.astype(float), marginal_hh_size.astype(float),
                             marginal_inc.astype(float), marginal_cfstat.astype(float)])

        # Try CO approach but not good...
        # p["result"] = comb_opti_integerization(p, total_pop)

        # probabilistic sampling
        p["result"] = probabilistic_sampling(p, total_pop)

        chunk = pd.DataFrame(
            columns=['sex', "prihm", "agegrp", "area", "hdgree", "lfact", "hhsize", 'totinc', 'cfstat'])
    else:
        print("Apply QISI (could be replaced by ipf bc much faster...)")
        p = humanleague.qisi(seed, [i0, i1, i2, i3, i4, i5, i6, i7],
                             [marginal_sex, marginal_prihm, marginal_age, marginal_age_by_sex, marginal_hdgree,
                              marginal_lfact, marginal_hh_size, marginal_inc])
        chunk = pd.DataFrame(columns=['sex', "prihm", "agegrp", "area", "hdgree", "lfact", "hhsize", 'totinc'])

    table = humanleague.flatten(p["result"])
    chunk.sex = table[0]
    chunk.prihm = table[1]
    chunk.agegrp = table[2]
    chunk.hdgree = table[3]
    chunk.lfact = table[4]
    chunk.hhsize = table[5]
    chunk.totinc = table[6]
    if fast:
        chunk.cfstat = table[7]
    chunk['area'] = int(DA_code)
    syn_inds = pd.concat([syn_inds, chunk], ignore_index=True)
    return syn_inds


# arg 1: path to files
# arg 2: province code (10, 11, 12, 13, 24, 35, 46, 47, 48, 59, 60, 61, 62)
# arg 3: from which DA code number start (generate 250 DAs). To generate all DAs in province, use -1
# arg 4: True or False, if True use IPF for fast population generation, else use QISI
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    province = str(sys.argv[2])
    from_DA = int(sys.argv[3])
    fast = bool(strtobool(sys.argv[4]))

    region = regions[province]

    df_indiv = load_indiv(path)
    DA_codes, filename = load_DAs(path)
    seed = load_seed(df_indiv, fast)
    census = load_census_profile(path, region)
    total_vb_id, total_age_by_sex_vb_id, total_hh_vb_id, age_vb, hdgree_vb, lfact_vb, hhsize_vb, totinc_vb, cfstat_vb = load_vbs_ids(
        census)
    province_census = census.loc[census['geocode'].astype(str) == province]

    print(province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0])
    syn_inds = pd.DataFrame(columns=['sex', "prihm", "agegrp", "area", "hdgree", "lfact", "hhsize", 'totinc'])
    if fast:
        syn_inds['cfstat'] = pd.Series(dtype='int')

    progress = from_DA + 1
    t0 = time.time()
    if from_DA == -1:
        from_DA = 0
        to_DA = len(DA_codes)
    else:
        to_DA = min(len(DA_codes), from_DA + 250)

    for DA_code in DA_codes[from_DA:to_DA]:
        print(str(progress) + "/" + str(to_DA))
        progress = progress + 1
        da_census = census.loc[census['geocode'] == DA_code]
        print("DA code: " + str(DA_code))
        #print(da_census.to_string())

        if not da_census.empty:
            if da_census.loc[da_census["variableId"] == total_vb_id]['total'].iloc[0] != '..':
                if int(da_census.loc[da_census["variableId"] == total_vb_id]['total'].iloc[0]) != 0:
                    syn_inds = synthetise_pop_da(syn_inds, DA_code, da_census, province_census, seed, fast)

    if not os.path.exists(path + "/" + filename + "/syn_pop"):
        os.makedirs(path + "/" + filename + "/syn_pop")
    syn_inds = syn_inds[['sex', 'prihm', 'agegrp', 'area', 'hdgree', 'lfact', 'hhsize', 'totinc']]
    syn_inds.to_csv(path + "/" + filename + "/syn_pop/synthetic_pop_" + str(to_DA) + ".csv", index=False)
    t1 = time.time()
    print(t1 - t0)
