import sys
import pandas as pd
import os


# Load synthetic population stats
def load_syn_pop(path, scenario):
    list_stats = []
    for file in os.listdir(path):
        if file.startswith("syn_pop_2021_stats_"+scenario+"_"):
            dat = pd.read_csv(path + "/" + file)
            list_stats.append(dat)
    df_stats = pd.concat(list_stats)
    df_stats= df_stats.drop_duplicates(subset=["area", "population", "households"]).reset_index(drop=True)

    return df_stats


# Load census stats
def load_census_stats(path):
    list_stats = []
    for file in os.listdir(path):
        if file.startswith("census_2021_stats_"):
            dat = pd.read_csv(path + "/" + file)
            list_stats.append(dat)
    df_stats = pd.concat(list_stats)
    df_stats['area'] = df_stats['area'].str.split('_').map(lambda x: '_'.join(sorted(x)))
    df_stats=df_stats.drop_duplicates(subset=["area", "population", "households"]).reset_index(drop=True)
    return df_stats


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Wrong number of arguments")
        sys.exit(1)
    path = sys.argv[1]

    for scenario in ['LG', 'M1', 'M2', 'M3', 'M4', 'M5', 'HG', 'SA', 'FA']:
        df_stats = load_syn_pop(path, scenario)
        df_stats.to_csv(path + "/syn_pop_2021_stats_"+scenario+".csv", index=False)

    df_census_stats = load_census_stats(path)
    df_census_stats.to_csv(path + "/census_2021_stats.csv", index=False)
