import sys
import pandas as pd
import os

# Load synthetic population for province
def load_syn_pop(path, filename):
    list_pop = []
    path = path + '/' + filename + '/syn_pop/'
    for file in os.listdir(path):
        if file.startswith("synthetic_pop_") & (not file.startswith("synthetic_pop_y"))& \
                (not file.endswith("hh.csv"))& (not file.endswith("hh_.csv")):
            dat = pd.read_csv(path + "/" + file)
            list_pop.append(dat)
    df_pop = pd.concat(list_pop)
    df_pop.reset_index(inplace=True)
    return df_pop


# Load synthetic population for province
def load_syn_pop_2016(path, filename):
    path = path + '/' + filename + '/syn_pop/synthetic_pop_y_2016.csv'
    df_pop = pd.read_csv(path)
    df_pop.reset_index(inplace=True)
    return df_pop


# Load population projections for province
def load_projections(path, scenario, year, province):
    file = path + '/projections_pop_17100057.csv'
    proj = pd.read_csv(file, usecols=['REF_DATE', 'GEO', 'DGUID', 'Projection scenario', 'Sex', 'Age group', 'VALUE'],
                       low_memory=False)
    province_proj = proj.loc[proj['DGUID'] == "2016A0002" + str(province)]
    province_proj = province_proj.loc[province_proj['Projection scenario'] == "Projection scenario " + scenario]
    province_proj = province_proj.loc[province_proj['REF_DATE'] == int(year)]
    province_proj['value'] = province_proj['VALUE'] * 1000
    return province_proj


# Load filename for province
def load_filename(path):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    lookup['pr'] = lookup[' PRuid/PRidu'].astype(str)
    filtered_lookup = lookup.loc[lookup['pr'].str.strip() == province]
    place = filtered_lookup.iloc[0][" PRename/PRanom"]
    print(place)
    filename = place.replace(" ", "_").lower()
    return filename


def get_projections_by_age_sex(province_proj):
    total_age_f = {}
    total_age_m = {}
    new_index = 0
    for i in range(0, 85, 5):
        total_age_m[new_index] = int(province_proj.loc[(province_proj["Sex"] == "Males") & (
                province_proj["Age group"] == str(i) + " to " + str(i + 4) + " years")]['value'].iloc[0])
        total_age_f[new_index] = int(province_proj.loc[(province_proj["Sex"] == "Females") & (
                province_proj["Age group"] == str(i) + " to " + str(i + 4) + " years")]['value'].iloc[0])
        new_index = new_index + 1

    total_age_m[new_index] = int(
        province_proj.loc[(province_proj["Sex"] == "Males") & (province_proj["Age group"] == "85 to 89 years")][
            'value'].iloc[0]) + int(
        province_proj.loc[(province_proj["Sex"] == "Males") & (province_proj["Age group"] == "90 to 94 years")][
            'value'].iloc[0]) + int(
        province_proj.loc[(province_proj["Sex"] == "Males") & (province_proj["Age group"] == "95 to 99 years")][
            'value'].iloc[0]) + int(province_proj.loc[(province_proj["Sex"] == "Males") & (
            province_proj["Age group"] == "100 years and over")][
                                        'value'].iloc[0])
    total_age_f[new_index] = int(
        province_proj.loc[(province_proj["Sex"] == "Females") & (province_proj["Age group"] == "85 to 89 years")][
            'value'].iloc[0]) + int(
        province_proj.loc[(province_proj["Sex"] == "Females") & (province_proj["Age group"] == "90 to 94 years")][
            'value'].iloc[0]) + int(
        province_proj.loc[(province_proj["Sex"] == "Females") & (province_proj["Age group"] == "95 to 99 years")][
            'value'].iloc[0]) + int(province_proj.loc[(province_proj["Sex"] == "Females") & (
            province_proj["Age group"] == "100 years and over")][
                                        'value'].iloc[0])
    return total_age_f, total_age_m


def get_missing_by_age_sex(totalAgeF, totalAgeM, df_pop, age_grps):
    missing_age_f = {}
    missing_age_m = {}
    for i in age_grps:
        missing_age_f[i] = totalAgeF[i] - len(df_pop.loc[(df_pop['sex'] == 0) & (df_pop['agegrp'] == i)].index)
        missing_age_m[i] = totalAgeM[i] - len(df_pop.loc[(df_pop['sex'] == 1) & (df_pop['agegrp'] == i)].index)
    return missing_age_f, missing_age_m


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    province = str(sys.argv[2])
    year = int(sys.argv[3])
    print(year)

    filename = load_filename(path)
    df_pop = load_syn_pop(path, filename)
    #df_pop = load_syn_pop_2016(path, filename)
    print(len(df_pop))

    if year == 2016:
        df_pop.to_csv(path + "/" + filename + "/syn_pop/synthetic_pop_y_" + str(year) + ".csv", index=False)

    elif (year > 2017) & (year < 2043):
        age_grps = df_pop['agegrp'].unique()
        age_grps.sort()

        province_proj = load_projections(path, "LG: low-growth", year, province)
        totalAgeF, totalAgeM = get_projections_by_age_sex(province_proj)
        missingAgeF, missingAgeM = get_missing_by_age_sex(totalAgeF, totalAgeM, df_pop, age_grps)

        for i in age_grps:
            if missingAgeF[i] > 0:
                if missingAgeF[i] > len(df_pop.loc[(df_pop['sex'] == 0) & (df_pop['agegrp'] == i)].index):
                    new = df_pop.loc[(df_pop['sex'] == 0) & (df_pop['agegrp'] == i)].sample(n=missingAgeF[i], replace=True)
                else:
                    new = df_pop.loc[(df_pop['sex'] == 0) & (df_pop['agegrp'] == i)].sample(n=missingAgeF[i])
                df_pop = pd.concat([df_pop, new])
            else:
                to_remove = df_pop.loc[(df_pop['sex'] == 0) & (df_pop['agegrp'] == i)].sample(n=-missingAgeF[i])
                df_pop = df_pop.drop(to_remove.index)

            if missingAgeM[i] > 0:
                if missingAgeM[i] > len(df_pop.loc[(df_pop['sex'] == 1) & (df_pop['agegrp'] == i)].index):
                    new = df_pop.loc[(df_pop['sex'] == 1) & (df_pop['agegrp'] == i)].sample(n=missingAgeM[i], replace=True)
                else:
                    new = df_pop.loc[(df_pop['sex'] == 1) & (df_pop['agegrp'] == i)].sample(n=missingAgeM[i])
                df_pop = pd.concat([df_pop, new])
            else:
                to_remove = df_pop.loc[(df_pop['sex'] == 1) & (df_pop['agegrp'] == i)].sample(n=-missingAgeM[i])
                df_pop = df_pop.drop(to_remove.index)
        print(len(df_pop.index))
        df_pop = df_pop[['sex', 'prihm', 'agegrp','area', 'hdgree', 'lfact', 'hhsize', 'totinc']]
        df_pop.to_csv(path + "/" + filename + "/syn_pop/synthetic_pop_y_" + str(year) + ".csv", index=False)
    else:
        print("Please indicate a year between 2018 and 2042")