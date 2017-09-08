import glob, os
import pandas as pd
from capstone_salad import rf_model_pipeline
from sklearn.model_selection import train_test_split
import csv


path = '/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14'

all_files = glob.glob(os.path.join(path, "*.zip"))
NJ_dfs = []

for file in all_files:
    NJ_dfs.append(pd.read_csv(file, header=None, error_bad_lines=False, dtype='object', quoting=csv.QUOTE_NONE, compression='zip'))


NJ = pd.concat(NJ_dfs)
NJ.reset_index(inplace=True, drop=True)

# Clean up some of the data
NJ['year'] = NJ[0].str[:4]
NJ['county'] = NJ[0].str[4:6]
NJ['municipality'] = NJ[0].str[6:8]
NJ['case_number'] = NJ[0].str[9:]
NJ['month'] = NJ[3].str[:2]
NJ['day_of_month'] = NJ[3].str[3:5]

NJ.to_csv('/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14/NJ_master.csv')

pd.read_csv('/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Occupants/occupants_subsample.csv'
