import sys
import pandas as pd
import pyreadstat
import math as m


# Load DA codes for CSD
def load_DAs(path, csd=""):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    if csd != "":
        lookup = lookup.loc[lookup[' CSDuid/SDRidu'].astype(str) == str(csd)]
    DA_codes = lookup[' DAuid/ADidu'].astype(str).unique()
    DA_codes.sort()
    print(str(DA_codes.size) + " DAs")

    return DA_codes


# Load provinces codes
def load_provinces(path):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    lookup['pr'] = lookup[' PRuid/PRidu'].astype(str)
    lookup[' PRename/PRanom'] = lookup[' PRename/PRanom'].apply(lambda x: x.lower())
    lookup[' PRename/PRanom'] = lookup[' PRename/PRanom'].apply(lambda x: x.replace(" ", "_"))
    provinces = dict(zip(lookup['pr'], lookup[' PRename/PRanom']))

    return provinces


# Load synthetic population for province
def load_syn_pop(path, year, filename, scenario):
    file = path + '/' + filename + '/syn_pop/' + scenario + '/synthetic_pop_' + year + '_hh_.csv'
    df_pop = pd.read_csv(file)
    df_pop['area'] = df_pop['area'].astype(str)
    return df_pop


# Load correspondence file 2021 - 2016
#https://www12.statcan.gc.ca/census-recensement/alternative_alternatif.cfm?l=eng&dispext=zip&teng=2021_92-156-X_DA_AD.zip&k=%20%20%20%20%203517&loc=//www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/correspondence-correspondance/files-fichiers/2021_92-156-X_DA_AD.zip
def load_correspondence_file(path):
    corresp = pd.read_csv(path + '/census_2021/2021_92-156-X_DA_AD.csv', encoding="ISO-8859-1", low_memory=False,
                       usecols=["DARELFLAG_ADINDREL","DADGUID2021_ADIDUGD2021","DADGUID2016_ADIDUGD2016"])
    corresp.rename(columns={'DARELFLAG_ADINDREL': 'flag',
                           'DADGUID2021_ADIDUGD2021': 'da21',
                           'DADGUID2016_ADIDUGD2016': 'da16'}, inplace=True)
    #print(corresp.loc[(corresp["flag"] == 4) & (corresp["da21"].str.startswith("2021S05124611"))]["da16"].unique())

    return corresp


# Load census 2021 profile
def load_census_profile(path, geocode):
    start_rows = pd.read_csv(
        path + '/census_2021/98-401-X2021006_eng_CSV/98-401-X2021006_Geo_starting_row.CSV', dtype=str,encoding='latin-1')
    index_start = start_rows.index[start_rows['Geo Code'] == geocode].tolist()[0]
    start = int(start_rows.loc[index_start]['Line Number'])
    end = int(start_rows.loc[index_start+1]['Line Number'])

    census = pd.read_csv(
        path + '/census_2021/98-401-X2021006_eng_CSV/98-401-X2021006_English_CSV_data.csv', encoding='latin-1',
        skiprows=range(1,start-1), nrows=end - start, low_memory=False,
        usecols=["DGUID",
                 "CHARACTERISTIC_ID",
                 "CHARACTERISTIC_NAME",
                 "C1_COUNT_TOTAL",
                 "C2_COUNT_MEN+",
                 "C3_COUNT_WOMEN+"
                 ]
    )
    census.rename(columns={'DGUID': 'geocode',
                           'CHARACTERISTIC_ID': 'variableId',
                           'CHARACTERISTIC_NAME': 'variable',
                           'C1_COUNT_TOTAL': 'total',
                           'C2_COUNT_MEN+': 'totalMale',
                           'C3_COUNT_WOMEN+': 'totalFemale'}, inplace=True)
    census = census.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return census


# Map sex to 2 classes
def map_sex(df_indiv):
    sexs = df_indiv["Sex"].unique()
    sexs.sort()
    index = 0
    for sex in sexs:
        df_indiv.loc[df_indiv["Sex"] == sex, "Sex"] = index
        index = index+1
    return df_indiv


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

    ages = df_indiv["agegrp"].unique()
    ages.sort()
    index = 0
    for age in ages:
        df_indiv.loc[df_indiv["agegrp"] == age, "agegrp"] = index
        index =  index+1
    return df_indiv


# Map hdgree to 3 classes
def map_hdgree(df_indiv):
    df_indiv.loc[df_indiv["hdgree"] == 88, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] == 99, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] > 2, "hdgree"] = 1686
    df_indiv.loc[df_indiv["hdgree"] == 1, "hdgree"] = 1684
    df_indiv.loc[df_indiv["hdgree"] == 2, "hdgree"] = 1685

    hdgrees = df_indiv["hdgree"].unique()
    hdgrees.sort()
    index = 0
    for hdgree in hdgrees:
        df_indiv.loc[df_indiv["hdgree"] == hdgree, "hdgree"] = index
        index = index + 1

    return df_indiv


# Map lfact to 3 classes
def map_lfact(df_indiv):
    df_indiv.loc[df_indiv["lfact"] == 1, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] == 2, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] < 11, "lfact"] = 1868
    df_indiv.loc[df_indiv["lfact"] < 100, "lfact"] = 1869

    lfacts = df_indiv["lfact"].unique()
    lfacts.sort()
    index = 0
    for lfact in lfacts:
        df_indiv.loc[df_indiv["lfact"] == lfact, "lfact"] = index
        index = index + 1
    return df_indiv


# Map hhsize to 5 classes
def map_hhsize(df_indiv):
    df_indiv.loc[df_indiv["hhsize"] == 8, "hhsize"] = 1
    df_indiv.loc[df_indiv["hhsize"] > 5, "hhsize"] = 5

    hhsizes = df_indiv["hhsize"].unique()
    hhsizes.sort()
    index = 0
    for hhsize in hhsizes:
        df_indiv.loc[df_indiv["hhsize"] == hhsize, "hhsize"] = index
        index = index + 1

    return df_indiv


# Map totinc to 4 classes
def map_totinc(df_indiv):
    df_indiv = df_indiv.loc[df_indiv["TotInc"] != 88888888]
    df_indiv.loc[df_indiv["TotInc"] == 99999999, "TotInc"] = 695
    df_indiv.loc[df_indiv["TotInc"] < 20000, "TotInc"] = 695

    df_indiv.loc[((df_indiv["TotInc"] >= 20000) & (df_indiv["TotInc"] < 60000)), "TotInc"] = 697
    df_indiv.loc[((df_indiv["TotInc"] >= 60000) & (df_indiv["TotInc"] < 100000)), "TotInc"] = 701
    df_indiv.loc[df_indiv["TotInc"] >= 100000, "TotInc"] = 705

    totincs = df_indiv["TotInc"].unique()
    totincs.sort()
    index = 0
    for totinc in totincs:
        df_indiv.loc[df_indiv["TotInc"] == totinc, "TotInc"] = index
        index = index + 1

    return df_indiv


# Map cfstat to 7 classes
def map_cfstat(df_indiv):
    df_indiv.loc[df_indiv["cfstat"] == 8, "cfstat"] = 7

    cfstats = df_indiv["cfstat"].unique()
    cfstats.sort()
    index = 0
    for cfstat in cfstats:
        df_indiv.loc[df_indiv["cfstat"] == cfstat, "cfstat"] = index
        index = index + 1

    return df_indiv


# Map hhtype to 5 classes
def map_hhtype(df_indiv):
    df_indiv.loc[df_indiv["cfstat"] == 0, "hhtype"] = 0
    df_indiv.loc[((df_indiv["cfstat"] == 1) | (df_indiv["cfstat"] == 3)), "hhtype"] = 1
    df_indiv.loc[((df_indiv["cfstat"] == 2) | (df_indiv["cfstat"] == 4)), "hhtype"] = 2
    df_indiv.loc[df_indiv["cfstat"] == 5, "hhtype"] = 3
    df_indiv.loc[df_indiv["cfstat"] == 6, "hhtype"] = 4

    return df_indiv


# Load individuals microdata
# Download from https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/GDJRT8
def load_indiv(path):
    dtafile = path + '/census_2016/PUMF/Census_2016_Individual_PUMF.dta'
    df_indiv, meta_indiv = pyreadstat.read_dta(dtafile, usecols=['ppsort', 'weight', 'agegrp', 'Sex',
                                                                 "hdgree", "lfact", 'TotInc',
                                                                 "hhsize", "cfstat", "prihm"])
    df_indiv = map_sex(df_indiv)
    df_indiv = map_age_grp(df_indiv)
    df_indiv = map_hdgree(df_indiv)
    df_indiv = map_lfact(df_indiv)
    df_indiv = map_hhsize(df_indiv)
    df_indiv = map_totinc(df_indiv)
    df_indiv = map_cfstat(df_indiv)
    df_indiv = map_hhtype(df_indiv)

    df_indiv = df_indiv.rename(columns={"Sex": "sex", "TotInc": "totinc"})

    return df_indiv


# Load variables identifiants in census
def load_vbs_ids(census):
    # Total population id
    total_vb_id = census.loc[census["variable"] == "Population, 2021"]['variableId'].iloc[0]

    # Total population sex id
    total_sex_vb_id = census.loc[census["variable"] == "Total - Age groups of the population - 100% data"]['variableId'].iloc[0]

    # Total population id in private dwellings
    total_pd_vb_id = census.loc[census["variable"] == "Total - Persons in private households - 100% data"]['variableId'].iloc[0]

    # Total households id
    total_hh_vb_id = \
        census.loc[census["variable"] == "Private dwellings occupied by usual residents"]['variableId'].iloc[0]

    # Total population density
    total_density_vb_id = census.loc[census["variable"] == "Population density per square kilometre"]['variableId'].iloc[0]

    # Total by age id
    age_vb = {}
    for i in range(0, 100, 5):
        age_vb[i] = census.loc[census["variable"] == str(i) + " to " + str(i + 4) + " years"]['variableId'].iloc[0]
    age_vb[100] = census.loc[census["variable"] == "100 years and over"]['variableId'].iloc[0]

    # Average population age
    avg_age_vb_id = census.loc[census["variable"] == "Average age of the population"]['variableId'].iloc[0]

    # Median population age
    med_age_vb_id = census.loc[census["variable"] == "Median age of the population"]['variableId'].iloc[0]

    # Total by hhsize id
    hhsize_vb = {}
    id_start = \
        census.loc[census["variable"] == "Total - Private households by household size - 100% data"]['variableId'].iloc[
            0] + 1
    for i in range(0, 5):
        hhsize_vb[i] = id_start + i

    # Average hh size
    avg_hhsize_vb_id = census.loc[census["variable"] == "Average household size"]['variableId'].iloc[0]

    # Total by totinc id
    totinc_vb = {}
    id_start = census.loc[census["variable"] ==
                          "Total - Total income groups in 2020 for the population aged 15 years and over in private " \
                          "households - 100% data"]['variableId'].iloc[0] + 3
    totinc_vb[0] = id_start
    totinc_vb[1] = id_start + 2
    totinc_vb[2] = id_start + 6
    totinc_vb[3] = id_start + 10
    totinc_vb[4] = id_start-1

    # Total by cfstat id
    cfstat_vb = {}
    cfstat_vb[0] = census.loc[census["variable"] == "Couple-family households"]['variableId'].iloc[0]+2
    cfstat_vb[1] = census.loc[census["variable"] == "Couple-family households"]['variableId'].iloc[0]+1
    cfstat_vb[2] = census.loc[census["variable"] == "One-parent-family households"]['variableId'].iloc[0]
    cfstat_vb[3] = census.loc[census["variable"] == "One-person households"]['variableId'].iloc[0]
    cfstat_vb[4] = census.loc[census["variable"] == "Total - Household type - 100% data"]['variableId'].iloc[0]
    return total_vb_id, total_sex_vb_id, total_pd_vb_id, total_hh_vb_id, total_density_vb_id, age_vb, avg_age_vb_id, med_age_vb_id, hhsize_vb, avg_hhsize_vb_id, totinc_vb, cfstat_vb


# Get stats total_pop, total_pop_pd, total_hh, avg_age, med_age and avg_hhsize
def get_totals_avg_med(census, df):
    area = census['geocode'].iloc[0]
    total_pop = census.loc[census["variableId"] == total_vb_id]['total'].iloc[0]
    total_pop_pd = census.loc[census["variableId"] == total_pd_vb_id]['total'].iloc[0]
    total_hh = census.loc[census["variableId"] == total_hh_vb_id]['total'].iloc[0]
    avg_age = census.loc[census["variableId"] == avg_age_vb_id]['total'].iloc[0]
    med_age = census.loc[census["variableId"] == med_age_vb_id]['total'].iloc[0]
    avg_hhsize = census.loc[census["variableId"] == avg_hhsize_vb_id]['total'].iloc[0]

    print(total_pop, "persons |", total_pop_pd, "persons in private dwellings |", total_hh, "households |", avg_age,
          "y-o in avg |", med_age, "y-o in median |", avg_hhsize, "avg hh size")

    df.loc[df['area'] == area, 'population'] = total_pop
    df.loc[df['area'] == area, 'population private dwellings'] = total_pop_pd
    df.loc[df['area'] == area, 'households'] = total_hh
    df.loc[df['area'] == area, 'avg age'] = avg_age
    df.loc[df['area'] == area, 'med age'] = med_age
    df.loc[df['area'] == area, 'avg hh size'] = avg_hhsize

    return df


# Get stats men and women
def get_totals_sex(census, df):
    area = census['geocode'].iloc[0]
    total_male = census.loc[census["variableId"] == total_sex_vb_id]['totalMale'].iloc[0]
    total_female = census.loc[census["variableId"] == total_sex_vb_id]['totalFemale'].iloc[0]

    print(total_male, "men |", total_female, "women")
    df.loc[df['area'] == area, 'males'] = total_male
    df.loc[df['area'] == area, 'females'] = total_female

    return df


# Get stats by age ranges
def get_ages(census, df):
    area = census['geocode'].iloc[0]
    total_age = {}
    for i in range(0, 101, 5):
        total_age[i] = census.loc[census["variableId"] == age_vb[i]]['total'].iloc[0]
        if i == 100:
            # print(i, 'and over', total_age[i], "persons")
            df.loc[df['area'] == area, '100+'] = total_age[i]
        else:
            # print(i, 'to', i + 4, total_age[i], "persons")
            df.loc[df['area'] == area, str(i)+'-'+str(i+4)] = total_age[i]
    return df


# Get stats by household size
def get_hhsizes(census, df):
    area = census['geocode'].iloc[0]
    total_hh_size = {}
    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = census.loc[census["variableId"] == hhsize_vb[i]]['total'].iloc[0]
        if i == len(hhsize_vb) - 1:
            df.loc[df['area'] == area, 'hh 5+p'] = total_hh_size[i]
            # print('Household with', i + 1, ' or more persons', total_hh_size[i])
        else:
            # print('Household with', i + 1, 'persons', total_hh_size[i])
            df.loc[df['area'] == area, 'hh '+str(i+1)+'p'] = total_hh_size[i]

    return df


# Get stats by income range
def get_incomes(census, df):
    area = census['geocode'].iloc[0]
    total_inc = {}
    if (census.loc[census["variableId"] == totinc_vb[0]]['total'].iloc[0] == "x") \
            or (census.loc[census["variableId"] == totinc_vb[0]]['total'].iloc[0] == "F") \
            or (m.isnan(census.loc[census["variableId"] == totinc_vb[0]]['total'].iloc[0])):
        total_inc[0] = "x"
        total_inc[1] = "x"
        total_inc[2] = "x"
        total_inc[3] = "x"
        total_inc[4] = "x"
    else:
        total_inc[0] = int(census.loc[census["variableId"] == totinc_vb[0]]['total'].iloc[0]) + \
                       int(census.loc[census["variableId"] == (totinc_vb[0] + 1)]['total'].iloc[0])
        for i in range(1, 3):
            total_inc[i] = int(census.loc[census["variableId"] == totinc_vb[i]]['total'].iloc[0]) + \
                           int(census.loc[census["variableId"] == (totinc_vb[i] + 1)]['total'].iloc[0]) + \
                           int(census.loc[census["variableId"] == (totinc_vb[i] + 2)]['total'].iloc[0]) + \
                           int(census.loc[census["variableId"] == (totinc_vb[i] + 3)]['total'].iloc[0])
        total_inc[3] = int(census.loc[census["variableId"] == totinc_vb[3]]['total'].iloc[0])
        total_inc[4] = int(census.loc[census["variableId"] == totinc_vb[4]]['total'].iloc[0])

    # print('Population aged 15+ with income < 20k$', total_inc[0])
    # print('Population aged 15+ with income 20k$<= and <60k$', total_inc[1])
    # print('Population aged 15+ with income 60k$<= and <100k$', total_inc[2])
    # print('Population aged 15+ with income >= 100k$', total_inc[3])
    df.loc[df['area'] == area, '15+ income < 20k$'] = total_inc[0]
    df.loc[df['area'] == area, '15+ income 20k$<= and <60k$'] = total_inc[1]
    df.loc[df['area'] == area, '15+ income 60k$<= and <100k$'] = total_inc[2]
    df.loc[df['area'] == area, '15+ income >= 100k$'] = total_inc[3]
    df.loc[df['area'] == area, '15+'] = total_inc[4]

    return df


# Get stats by household type
def get_hhtypes(census, df):
    area = census['geocode'].iloc[0]
    total_cfstat = {}

    if (census.loc[census["variableId"] == cfstat_vb[0]]['total'].iloc[0] == "x") \
            or (census.loc[census["variableId"] == cfstat_vb[0]]['total'].iloc[0] == "F") \
            or (m.isnan(census.loc[census["variableId"] == cfstat_vb[0]]['total'].iloc[0])):
        total_cfstat[0] = "x"
        total_cfstat[1] = "x"
        total_cfstat[2] = "x"
        total_cfstat[3] = "x"
        total_cfstat[4] = "x"
    else:
        for i in range(0, 5):
            total_cfstat[i] = int(census.loc[census["variableId"] == cfstat_vb[i]]['total'].iloc[0])
        total_cfstat[4] = total_cfstat[4] - (total_cfstat[0] + total_cfstat[1] + total_cfstat[2] + total_cfstat[3])
        #print('Couples without children', total_cfstat[0], 'households')
        #print('Couples with children', total_cfstat[1], 'households')
        #print('One-parent-family', total_cfstat[2], 'households')
        #print('One-person', total_cfstat[3], 'households')
        #print('Other kind of household',
        #      total_cfstat[4] - (total_cfstat[0] + total_cfstat[1] + total_cfstat[2] + total_cfstat[3]), 'households')

    df.loc[df['area'] == area, 'Couples without children'] = total_cfstat[0]
    df.loc[df['area'] == area, 'Couples with children'] = total_cfstat[1]
    df.loc[df['area'] == area, 'One-parent-family'] = total_cfstat[2]
    df.loc[df['area'] == area, 'One-person'] = total_cfstat[3]
    df.loc[df['area'] == area, 'Other kind of hh'] = total_cfstat[4]
    return df


# Compute all stats for the synthetic population in the given area
def compute_stats_synth_pop(df_pop, df, area):
    df = pd.concat([pd.DataFrame({'area': area}, index=[0]), df], ignore_index=True)

    total_pop = len(df_pop.index)
    total_males = len(df_pop.loc[df_pop['sex'] == 1].index)
    total_females = len(df_pop.loc[df_pop['sex'] == 0].index)
    total_pop_pd = len(df_pop.loc[df_pop["HID"] != -1].index)
    total_hh = len(df_pop.loc[(df_pop["HID"] != -1) & (df_pop["prihm"] == 1)].index)
    avg_age = round(df_pop['age'].mean(), 1)
    med_age = round(df_pop['age'].median(), 1)
    avg_hhsize = round(df_pop['hhsize'].mean(), 1)

    # print(total_pop, "persons |", total_males, "men |", total_females, "women |", total_pop_pd,
    # "persons in private dwellings |", total_hh, "households |", avg_age, "y-o in avg |", med_age, "y-o in median |",
    # avg_hhsize, "avg hh size")
    df.loc[df['area'] == area, 'population'] = total_pop

    df.loc[df['area'] == area, 'males'] = total_males
    df.loc[df['area'] == area, 'females'] = total_females
    df.loc[df['area'] == area, 'population private dwellings'] = total_pop_pd
    df.loc[df['area'] == area, 'households'] = total_hh
    df.loc[df['area'] == area, 'avg age'] = avg_age
    df.loc[df['area'] == area, 'med age'] = med_age
    df.loc[df['area'] == area, 'avg hh size'] = avg_hhsize

    total_age = {}
    for age in range(0, 101, 5):
        total_age[age] = len(df_pop.loc[(df_pop["age"] >= age) & (df_pop["age"] < age+5)].index)
        if age == 100:
            # print(age, 'and over', total_age[age], "persons")
            df.loc[df['area'] == area, '100+'] = total_age[age]
        else:
            # print(age, 'to', age + 4, total_age[age], "persons")
            df.loc[df['area'] == area, str(age) + '-' + str(age + 4)] = total_age[age]

    total_hh_size = {}
    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = len(df_pop.loc[(df_pop["HID"] != -1) & (df_pop["prihm"] == 1) &
                                          (df_pop["hhsize"] == i)].index)
        if i == len(hhsize_vb) - 1:
            # print('Household with', i + 1, ' or more persons', total_hh_size[i])
            df.loc[df['area'] == area, 'hh 5+p'] = total_hh_size[i]
        else:
            # print('Household with', i + 1, 'persons', total_hh_size[i])
            df.loc[df['area'] == area, 'hh ' + str(i + 1) + 'p'] = total_hh_size[i]

    total_inc = {}
    for i in range(0, 4):
        total_inc[i] = len(df_pop.loc[(df_pop["HID"] != -1) & (df_pop["totinc"] == i) & (df_pop["age"] >= 15)].index)
    # print('Population aged 15+ with income < 20k$', total_inc[0])
    # print('Population aged 15+ with income 20k$<= and <60k$', total_inc[1])
    # print('Population aged 15+ with income 60k$<= and <100k$', total_inc[2])
    # print('Population aged 15+ with income >= 100k$', total_inc[3])
    df.loc[df['area'] == area, '15+'] = len(df_pop.loc[df_pop["age"] >= 15].index)
    df.loc[df['area'] == area, '15+ income < 20k$'] = total_inc[0]
    df.loc[df['area'] == area, '15+ income 20k$<= and <60k$'] = total_inc[1]
    df.loc[df['area'] == area, '15+ income 60k$<= and <100k$'] = total_inc[2]
    df.loc[df['area'] == area, '15+ income >= 100k$'] = total_inc[3]

    total_cfstat = {}
    for i in range(0, 5):
        total_cfstat[i] = len(df_pop.loc[(df_pop["HID"] != -1) & (df_pop["prihm"] == 1) &
                                         (df_pop["hhtype"] == i)].index)
    # print('Couples with children', total_cfstat[0], 'households')
    # print('Couples without children', total_cfstat[1], 'households')
    # print('One-parent-family', total_cfstat[2], 'households')
    # print('One-person', total_cfstat[3], 'households')
    # print('Other kind of household', total_cfstat[4], 'households')
    df.loc[df['area'] == area, 'Couples without children'] = total_cfstat[0]
    df.loc[df['area'] == area, 'Couples with children'] = total_cfstat[1]
    df.loc[df['area'] == area, 'One-parent-family'] = total_cfstat[2]
    df.loc[df['area'] == area, 'One-person'] = total_cfstat[3]
    df.loc[df['area'] == area, 'Other kind of hh'] = total_cfstat[4]

    df_private_dwellings = df_pop.loc[df_pop["HID"] != -1]
    df_unique_indivs = df_private_dwellings.groupby(
        ['agegrp', 'hhtype', 'hdgree', 'hhsize', 'lfact', 'prihm', 'sex', 'totinc']).size().reset_index(name='Count')
    df_merge = pd.merge(df_unique_indivs, unique_indivs,
                        on=['agegrp', 'hhtype', 'hdgree', 'hhsize', 'lfact', 'prihm', 'sex', 'totinc'], how='inner')
    # print(sum(df_merge["Count_x"]), " out of ", len(df_private_dwellings.index), 'individuals are from PUMF')
    # print((sum(df_merge["Count_x"]) / len(df_private_dwellings.index)) * 100.0, " % are realistic")
    df.loc[df['area'] == area, 'realistic individuals'] = sum(df_merge["Count_x"])
    if len(df_private_dwellings.index)!=0:
        df.loc[df['area'] == area, '% realistic individuals'] = (sum(df_merge["Count_x"]) / total_pop_pd) * 100.0
    else:
        df.loc[df['area'] == area, '% realistic individuals'] = 0

    return df


# Get all stats from the given census area
def get_stats_census(census21, df):
    df = pd.concat([pd.DataFrame({'area': census21['geocode'].iloc[0]}, index=[0]), df], ignore_index=True)
    df = get_totals_avg_med(census21, df)
    df = get_totals_sex(census21, df)
    df = get_ages(census21, df)
    df = get_hhsizes(census21, df)
    df = get_incomes(census21, df)
    df = get_hhtypes(census21, df)
    return df


# Get stats census for Canada, provinces and given city and save to file
def generate_stats_census(path, year, canada_census21, provinces, city, from_):
    # Stats for Canada
    print("Canada")
    df_stats_census = pd.DataFrame(
        columns=['area', 'population', 'males', 'females', 'population private dwellings', 'households', 'avg age',
                 'med age', 'avg hh size', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                 '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74',
                 '75-79', '80-84', '85-89', '90-94', '95-99', '100+',
                 'hh 1p', 'hh 2p', 'hh 3p', 'hh 4p', 'hh 5+p',
                 '15+ income < 20k$', '15+ income 20k$<= and <60k$',
                 '15+ income 60k$<= and <100k$', '15+ income >= 100k$', '15+',
                 'Couples without children', 'Couples with children', 'One-parent-family',
                 'One-person', 'Other kind of hh'
                 ])
    canada_census21["geocode"] = canada_census21["geocode"].str.replace('2021A','2016A')
    df_stats_census = get_stats_census(canada_census21, df_stats_census)

    # Stats for Provinces
    census21 = {}
    for pr in provinces.keys():
        print("Province", pr)
        census21[pr] = load_census_profile(path, '2021A0002' + str(pr))
        census21[pr]["geocode"] = census21[pr]["geocode"].str.replace('2021A', '2016A')
        df_stats_census = get_stats_census(census21[pr], df_stats_census)

    # Stats for cities
    # cities = [4611040, 3520005, 2443027]
    cities = [city]
    for csd in cities:
        print("City", csd)
        census21[csd] = load_census_profile(path, '2021A0005' + str(csd))
        census21[csd]["geocode"] = census21[csd]["geocode"].str.replace('2021A', '2016A')
        df_stats_census = get_stats_census(census21[csd], df_stats_census)

    # Stats for DAs
    corresp = load_correspondence_file(path)
    DA_codes = load_DAs(path)
    if from_ == -1:
        from_DA = 0
        to_DA = len(DA_codes)
    else:
        from_DA = from_
        to_DA = min(len(DA_codes), from_DA + 2000)
    cpt = 1
    for da in DA_codes[from_DA:to_DA]:
        print(cpt, "/", len(DA_codes))
        da16 = '2016S0512' + da
        flag = corresp.loc[(corresp["da16"] == da16)]["flag"].unique()[0]
        if flag == 1:
            da21 = corresp.loc[(corresp["da16"] == da16)]["da21"].unique()[0]
            census = load_census_profile(path, da21)
            census.loc[census["geocode"] == da21, "geocode"] = da16
            df_stats_census = get_stats_census(census, df_stats_census)
        elif flag == 3:
            das21 = corresp.loc[(corresp["da16"] == da16)]["da21"].unique()
            for da21 in das21:
                census = load_census_profile(path, da21)
                df_stats_census = get_stats_census(census, df_stats_census)
            df_stats_census.loc[df_stats_census["area"].isin(das21), "area"] = da16
        elif (flag == 4) | (flag == 2):
            das16 = [da16]
            das21 = corresp.loc[(corresp["da16"] == da16)]["da21"].unique().tolist()
            olddas21 = []
            while(len(das21) > len(olddas21)):
                olddas21 = das21
                for da21 in das21:
                    das16.extend(corresp.loc[(corresp["da21"] == da21)]["da16"].unique().tolist())
                    das16 = list(set(das16))
                    for da16 in das16:
                        das21.extend(corresp.loc[(corresp["da16"] == da16)]["da21"].unique().tolist())
                        das21 = list(set(das21))
            for da21 in das21:
                census = load_census_profile(path, da21)
                df_stats_census = get_stats_census(census, df_stats_census)
            df_stats_census.loc[df_stats_census["area"].isin(das21), "area"] = '_'.join(map(str, das16))
        cpt = cpt + 1
    df_stats_census.to_csv(path + '/census_' + year + '_stats_' + str(city) + '_' + str(to_DA) + '.csv', index=False)


# Get stats synthetic population for Canada, provinces and given city and save to file
def generate_stats_syn_pop(path, year, provinces, city, from_, scenario):
    df_stats_syn_pop = pd.DataFrame(
                columns=['area', 'population', 'males', "females",'population private dwellings', 'households', 'avg age',
                 'med age', 'avg hh size', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                 '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74',
                 '75-79', '80-84', '85-89', '90-94', '95-99', '100+',
                 'hh 1p', 'hh 2p', 'hh 3p', 'hh 4p', 'hh 5+p',
                 '15+ income < 20k$', '15+ income 20k$<= and <60k$',
                 '15+ income 60k$<= and <100k$', '15+ income >= 100k$', '15+',
                 'Couples without children', 'Couples with children', 'One-parent-family',
                 'One-person', 'Other kind of hh',
                 'realistic individuals', '% realistic individuals'
                 ])

    list_pop = []
    for pr in provinces.values():
        print("Province", pr)
        df_pr = load_syn_pop(path, year, pr, scenario)
        list_pop.append(df_pr)
        df_stats_syn_pop = compute_stats_synth_pop(df_pr, df_stats_syn_pop,
                                                '2016A0002'+list(provinces.keys())[list(provinces.values()).index(pr)])
    df_pop = pd.concat(list_pop)
    df_pop.reset_index(inplace=True)
    print("Canada")
    df_stats_syn_pop = compute_stats_synth_pop(df_pop, df_stats_syn_pop, '2016A000011124')

    #cities = [4611040, 3520005, 2443027]
    #cities = [city]
    #for csd in cities:
        #print("City", csd)
    DA_codes = load_DAs(path)
    if from_ == -1:
        from_DA = 0
        to_DA = len(DA_codes)
    else:
        from_DA = from_
        to_DA = min(len(DA_codes), from_DA + 2000)
    cpt = 1
    # for da in DA_codes:
    for da in DA_codes[from_DA:to_DA]:
        print(cpt, "/", len(DA_codes))
        df_stats_syn_pop = compute_stats_synth_pop(df_pop.loc[(df_pop.area == da)], df_stats_syn_pop, '2016S0512' + da)
        cpt = cpt + 1

    df_stats_syn_pop.to_csv(path + '/syn_pop_' + year + '_stats_'+ scenario+'_'+ str(city) + '_' + str(to_DA) + '.csv', index=False)


# Path: path to files
# city: csd code of the city of interest
# from_DA:  -1 to generate all DAs of the city.
#           If parallelization: DAs are generated 2000 by 2000, need to give the starting DA number
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    city = sys.argv[2]
    from_DA = int(sys.argv[3])
    year = "2021"
    print(year)
    scenario = sys.argv[4]

    provinces = load_provinces(path)
    df_indiv = load_indiv(path)
    unique_indivs = df_indiv.groupby(['agegrp', 'hhtype', 'hdgree', 'hhsize',
                                      'lfact','prihm', 'sex', 'totinc']).size().reset_index(name='Count')

    canada_census21 = load_census_profile(path, '2021A000011124')
    total_vb_id, total_sex_vb_id, total_pd_vb_id, total_hh_vb_id, total_density_vb_id, age_vb, avg_age_vb_id, \
    med_age_vb_id, hhsize_vb, avg_hhsize_vb_id, totinc_vb, cfstat_vb = load_vbs_ids(canada_census21)

    #generate_stats_census(path, year, canada_census21, provinces, city, from_DA)
    generate_stats_syn_pop(path, year, provinces, city, from_DA, scenario)
