import glob, os
import pandas as pd
from capstone_salad import rf_model_pipeline
from sklearn.model_selection import train_test_split
import csv


path = '/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Occupants/'

all_files = glob.glob(os.path.join(path, "*.zip"))
NJ_dfs = []

for my_file in all_files:
    NJ_dfs.append(pd.read_csv(my_file, header=None, error_bad_lines=False, dtype='object'))

all_files_txt = glob.glob(os.path.join(path, "*.txt"))
for my_file in all_files_txt:
    NJ_dfs.append(pd.read_csv(my_file, header=None, error_bad_lines=False, dtype='object'))

NJ = pd.concat(NJ_dfs)
NJ.reset_index(inplace=True, drop=True)


NJ.to_csv('/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Occupants/NJ_occupants.csv')
