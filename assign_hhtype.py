import sys
import pandas as pd


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


# Load synthetic population for province
def load_syn_pop(path, year, filename, scenario):
    if year == "2016":
        path = path + '/' + filename + '/syn_pop/'
    else:
        path = path + '/' + filename + '/syn_pop/' + scenario + '/'
    file = path + 'synthetic_pop_' + year + '_hh.csv'
    df_pop = pd.read_csv(file)
    df_pop['area'] = df_pop['area'].astype(str)
    print(len(df_pop.index))

    return df_pop


def compute_hhtypes(df_pop):
    df_pop['hhtype'] = -1
    for hh_id in df_pop.loc[df_pop["HID"] != -1]["HID"].unique():
        hh = df_pop.loc[df_pop["HID"] == hh_id]
        if len(hh.index) == 1:
            df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 3  # one person hh
        elif len(hh.index) == 2:
            if (abs(hh.iloc[0]['age'] - hh.iloc[1]['age']) > 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 2  # one-parent family
            elif (hh.iloc[0]['age'] > 16) & (hh.iloc[1]['age'] > 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 0  # couple without children
        elif len(hh.index) == 3:
            ages = hh['age'].to_list()
            ages.sort()
            youngest = ages[0]
            oldest_1 = ages[1]
            oldest_2 = ages[2]
            if ((youngest<16) & (oldest_1>16) & (oldest_2>16)) | ((oldest_1 - youngest > 16) & (oldest_2 - youngest > 16)):  # & (oldest_2 - oldest_1 < 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 1  # couple with children
            elif (youngest < 16) & (oldest_1 < 16) & (oldest_2 - youngest > 16) & (oldest_2 - oldest_1 > 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 2  # one-parent family
        elif len(hh.index) == 4:
            ages = hh['age'].to_list()
            ages.sort()
            youngest_1 = ages[0]
            youngest_2 = ages[1]
            oldest_1 = ages[2]
            oldest_2 = ages[3]
            if ((youngest_1<16) & (youngest_2<16) & (oldest_1>16) & (oldest_2>16)) |\
                    ((oldest_1 - youngest_1 > 16) & (oldest_1 - youngest_2 > 16) & (oldest_2 - youngest_1 > 16) & (
                    oldest_2 - youngest_2 > 16)):  # & (oldest_2 - oldest_1 < 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 1  # couple with children
            elif (youngest_1 < 16) & (youngest_2 < 16) & (oldest_1 < 16) & (oldest_2 - youngest_1 > 16) & (
                    oldest_2 - youngest_2 > 16) & (oldest_2 - oldest_1 > 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 2  # one-parent family
        elif len(hh.index) == 5:
            ages = hh['age'].to_list()
            ages.sort()
            youngest_1 = ages[0]
            youngest_2 = ages[1]
            youngest_3 = ages[2]
            oldest_1 = ages[3]
            oldest_2 = ages[4]
            if ((youngest_1<16) & (youngest_2<16) & (youngest_3<16) & (oldest_1>16) & (oldest_2>16)) | \
            ((oldest_1 - youngest_1 > 16) & (oldest_1 - youngest_2 > 16) & (oldest_1 - youngest_3 > 16) & (
                    oldest_2 - youngest_1 > 16) & (oldest_2 - youngest_2 > 16) & (
                    oldest_2 - youngest_3 > 16)):  # & (oldest_2 - oldest_1 < 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 1  # couple with children
            elif (youngest_1 < 16) & (youngest_2 < 16) & (youngest_3 < 16) & (oldest_1 < 16) & (
                    oldest_2 - youngest_1 > 16) & (oldest_2 - youngest_2 > 16) & (oldest_2 - youngest_3 > 16) & (
                    oldest_2 - oldest_1 > 16):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 2  # one-parent family
        elif len(hh.index) == 6:
            ages = hh['age'].to_list()
            ages.sort()
            youngest_1 = ages[0]
            youngest_2 = ages[1]
            youngest_3 = ages[2]
            youngest_4 = ages[3]
            oldest_1 = ages[4]
            oldest_2 = ages[5]
            if ((youngest_1<16)&(youngest_2<16)&(youngest_3<16)&(youngest_4<16)&(oldest_1>16)&(oldest_2>16)) | ((oldest_1 - youngest_1 > 16) & (oldest_1 - youngest_2 > 16) & (oldest_1 - youngest_3 > 16) & (
                    oldest_1 - youngest_4 > 16) & \
                    (oldest_2 - youngest_1 > 16) & (oldest_2 - youngest_2 > 16) & (oldest_2 - youngest_3 > 16) & (
                    oldest_2 - youngest_4 > 16)):  # & (oldest_2 - oldest_1 < 16)):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 1  # couple with children
            elif ((youngest_1 < 16) & (youngest_2 < 16) & (youngest_3 < 16) & (youngest_4 < 16) & (oldest_1 < 16) & (
                    oldest_2 - youngest_1 > 16) & (oldest_2 - youngest_2 > 16) & (oldest_2 - youngest_3 > 16) & (
                          oldest_2 - youngest_4 > 16) & (
                          oldest_2 - oldest_1 > 16)):
                df_pop.loc[df_pop["HID"] == hh_id, 'hhtype'] = 2  # one-parent family

    df_pop.loc[(df_pop["HID"] != -1) & (df_pop["hhtype"] == -1), 'hhtype'] = 4  # other kind
    return df_pop


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    province = str(sys.argv[2])
    from_DA = int(sys.argv[3])
    year = sys.argv[4]
    scenario = sys.argv[5]
    print(year)

    DA_codes, filename = load_DAs(path)
    df_indivs = load_syn_pop(path, year, filename, scenario)

    progress = from_DA + 1
    if from_DA == -1:
        from_DA = 0
        to_DA = len(DA_codes)
    else:
        to_DA = min(len(DA_codes), from_DA + 1000)

    df_indivs = df_indivs[df_indivs['area'].isin(DA_codes[from_DA:to_DA])]
    df_indivs = compute_hhtypes(df_indivs)

    if year == "2016":
        output_path = path + '/' + filename + '/syn_pop'
    else:
        output_path = path + "/" + filename + '/syn_pop/' + scenario

    if not df_indivs.empty:
        df_indivs = df_indivs[
            ['HID', 'sex', 'prihm', 'agegrp', 'age', 'area', 'hdgree', 'lfact', 'hhsize', 'totinc', 'hhtype']]
        if (from_DA == 0) & (to_DA == len(DA_codes)):
            df_indivs.to_csv(output_path + "/synthetic_pop_" + str(year) + "_hh_.csv", index=False)
        else:
            df_indivs.to_csv(output_path + "/synthetic_pop_" + str(year) + "_" + str(to_DA) + "_hh_.csv", index=False)