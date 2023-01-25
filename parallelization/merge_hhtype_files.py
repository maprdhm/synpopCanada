import sys
import pandas as pd
import os


# Load synthetic population for province
def load_syn_pop(path, filename, scenario):
    df_pop = pd.DataFrame()
    list_pop = []
    if year == "2016":
        path = path + '/' + filename + '/syn_pop/'
    else:
        path = path + '/' + filename + '/syn_pop/' + scenario + '/'

    for file in os.listdir(path):
        if file.startswith("synthetic_pop_"+str(year)) & (file.endswith("hh_.csv") & (file != "synthetic_pop_"+str(year)+"_hh_.csv")):
            dat = pd.read_csv(path + "/" + file)
            list_pop.append(dat)
    if len(list_pop) != 0:
        df_pop = pd.concat(list_pop)

    return df_pop


# Load DA codes for province
def load_DAs(path,province):
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


def merge(path, province, year, scenario):
    DA_codes, filename = load_DAs(path, province)

    df_pop = load_syn_pop(path, filename, scenario)
    if not df_pop.empty:
        print(len(df_pop))

        df_pop = df_pop.drop(['level_0'], axis=1, errors='ignore')
        df_pop = df_pop.drop(['Unnamed: 0'], axis=1, errors='ignore')

        if year == "2016":
            output_path = path + '/' + filename + '/syn_pop'
        else:
            output_path = path + "/" + filename + "/syn_pop/" + scenario
        df_pop.to_csv(output_path + "/synthetic_pop_" + str(year) + "_hh_.csv", index=False)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Wrong number of arguments")
        sys.exit(1)

    path = sys.argv[1]
    province = str(sys.argv[2])
    year = int(sys.argv[3])
    scenario = str(sys.argv[4])

    if (province == "-1") & (year == -1) & (scenario == "-1"):
        for province in ['10', '11', '12', '13', '24', '35', '46', '47', '48', '59', '60', '61', '62']:
            for year in ['2016', '2021', '2023', '2030']:
                for scenario in ['LG', 'M1', 'M2', 'M3', 'M4', 'M5', 'HG', 'SA', 'FA']:
                    merge(path, province, year, scenario)
    else:
        merge(path, province, year, scenario)
