import sys
import pandas as pd
import numpy as np
import os

# Load synthetic population for province
def load_syn_pop(path, filename, year):
    list_pop = []
    hid_index = 0
    path = path + '/' + filename + '/syn_pop/'
    for file in os.listdir(path):
        if file.startswith("synthetic_pop_"+str(year)) & (file.endswith("hh.csv") & (file != "synthetic_pop_"+str(year)+"_hh.csv")):
            dat = pd.read_csv(path + "/" + file)
            dat['HID'] =dat['HID'].astype(int)
            print(len(dat.loc[dat['HID'] == -1].index))
            print(len(dat.loc[dat['HID'] != -1].index))

            hids_list = dat['HID'].unique()
            hids_list.sort()
            hids_list = np.delete(hids_list, np.where(hids_list == -1))
            test_values = range(hid_index, hid_index+len(hids_list))
            map_dict = {hids_list[i]: test_values[i] for i in range(len(hids_list))}
            map_dict[-1] = -1
            hid_index = int(hid_index+len(hids_list))

            dat['HID'] = dat['HID'].map(map_dict)

            list_pop.append(dat)
    df_pop = pd.concat(list_pop)
    print(df_pop)
    df_pop.reset_index(inplace=True, drop=True)
    return df_pop


# Load DA codes for province
def load_DAs(path, province):
    lookup = pd.read_csv(path + '/census_2016/lookup.csv', encoding="ISO-8859-1", low_memory=False)
    lookup['pr'] = lookup[' PRuid/PRidu'].astype(str)
    filtered_lookup = lookup.loc[lookup['pr'].str.strip() == str(province)]
    place = filtered_lookup.iloc[0][" PRename/PRanom"]
    print(place)
    filename = place.replace(" ", "_").lower()
    DA_codes = filtered_lookup[' DAuid/ADidu'].unique()
    DA_codes.sort()
    print(str(DA_codes.size) + " DAs")
    return DA_codes, filename


def merge(path, province, year):
    DA_codes, filename = load_DAs(path, province)
    df_pop = load_syn_pop(path, filename, year)
    print(len(df_pop))
    df_pop = df_pop.drop(['level_0'], axis=1, errors='ignore')
    df_pop = df_pop.drop(['Unnamed: 0'], axis=1, errors='ignore')
    if not df_pop.empty:
        df_pop.to_csv(path + "/" + filename + "/syn_pop/synthetic_pop_" + str(year) + "_hh.csv", index=False)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]
    province = str(sys.argv[2])
    year = int(sys.argv[3])
    print(year)

    if (province == "-1") & (year == -1):
        for province in ['10','11','12','13','24','35','46','47','48','59','60', '61', '62']:
            for year in ['2016', '2021', '2022', '2030']:
                merge(path, province, year)
    else:
        merge(path, province, year)
