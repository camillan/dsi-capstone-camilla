import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from capstone_salad import cols_not_shared, rf_model_pipeline, max_by_group, mean_by_group, min_by_group

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Loading the first set of CSVs
accidents = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/accident.csv')
persons = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/person.csv')
vehicles = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/vehicle.csv')
maneuvers = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/maneuver.csv')
violations = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/violatn.csv')
pbtype = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/pbtype.csv')

accidents.columns
accidents.isnull().sum()


# We see that 'TWAY_ID2' is the only one that has a lot of missing values. As it's not an important variable, we drop it. We also drop rail column because it seems to be poor data.
accidents.drop('TWAY_ID', axis=1, inplace=True)
accidents.drop('TWAY_ID2', axis=1, inplace=True)
accidents.drop('RAIL', axis=1, inplace=True)
# We also need to modify a certain columns: the ALC_RES, to make values over 939 into 0 (see data dictionary)
persons.ALC_RES.replace(to_replace=range(939, 1000), value=0, inplace=True)


# Set the ST_CASE (the unique case identifier) as the index.
accidents.set_index('ST_CASE', inplace=True)


# First, we start with the persons DF. We want to see the columns that are in persons but not in accidents.
columns_onlyin_persons = cols_not_shared(accidents, persons)
cropped_persons = persons[columns_onlyin_persons]


# For some classed variables, we want dummies. We'll then groupby ST_CASE and put into the max function.
cols_for_dummifying = ['REST_USE', 'REST_MIS', 'AIR_BAG', 'EJECTION', 'EJ_PATH', 'P_SF1', 'P_SF2', 'P_SF3']
dummied_persons = pd.get_dummies(cropped_persons, columns=cols_for_dummifying)
cols_after_dummying = ['REST_USE_0', 'REST_USE_1', 'REST_USE_2', 'REST_USE_3', 'REST_USE_4', 'REST_USE_5', 'REST_USE_7', 'REST_USE_8', 'REST_USE_10', \
 'REST_USE_11', 'REST_USE_12', 'REST_USE_16', \
       'REST_USE_17', 'REST_USE_19', 'REST_USE_29', 'REST_USE_96', \
       'REST_USE_97', 'REST_USE_98', 'REST_USE_99', 'REST_MIS_0', \
       'REST_MIS_1', 'REST_MIS_8', 'AIR_BAG_0', 'AIR_BAG_1', 'AIR_BAG_2', \
       'AIR_BAG_3', 'AIR_BAG_7', 'AIR_BAG_8', 'AIR_BAG_9', 'AIR_BAG_20', \
       'AIR_BAG_28', 'AIR_BAG_97', 'AIR_BAG_98', 'AIR_BAG_99', \
       'EJECTION_0', 'EJECTION_1', 'EJECTION_2', 'EJECTION_3', \
       'EJECTION_7', 'EJECTION_8', 'EJECTION_9', 'EJ_PATH_0', 'EJ_PATH_1', \
       'EJ_PATH_2', 'EJ_PATH_3', 'EJ_PATH_4', 'EJ_PATH_5', 'EJ_PATH_6', \
       'EJ_PATH_7', 'EJ_PATH_8', 'EJ_PATH_9', 'P_SF1_0', 'P_SF1_5', \
       'P_SF1_8', 'P_SF1_9', 'P_SF1_13', 'P_SF1_18', 'P_SF1_21', \
       'P_SF1_32', 'P_SF1_37', 'P_SF1_42', 'P_SF1_56', 'P_SF1_60', \
       'P_SF1_61', 'P_SF1_62', 'P_SF1_64', 'P_SF1_65', 'P_SF1_66', \
       'P_SF1_68', 'P_SF1_72', 'P_SF1_75', 'P_SF1_76', 'P_SF1_80', \
       'P_SF1_82', 'P_SF1_86', 'P_SF1_87', 'P_SF1_88', 'P_SF1_89', \
       'P_SF1_90', 'P_SF1_91', 'P_SF1_92', 'P_SF1_99', 'P_SF2_0', \
       'P_SF2_5', 'P_SF2_66', 'P_SF2_82', 'P_SF2_99', 'P_SF3_0', \
       'P_SF3_99']



# Append it with
all_cols_for_max = ['REST_USE_0', 'REST_USE_1', 'REST_USE_2', 'REST_USE_3', 'REST_USE_4', 'REST_USE_5', 'REST_USE_7', 'REST_USE_8', 'REST_USE_10', 'REST_USE_11', 'REST_USE_12', 'REST_USE_16', 'REST_USE_17', 'REST_USE_19', 'REST_USE_29', 'REST_USE_96', 'REST_USE_97', 'REST_USE_98', 'REST_USE_99', 'REST_MIS_0', "REST_MIS_1", 'REST_MIS_8', 'AIR_BAG_0', 'AIR_BAG_1', 'AIR_BAG_2', 'AIR_BAG_3', 'AIR_BAG_7', 'AIR_BAG_8', 'AIR_BAG_9', 'AIR_BAG_20', 'AIR_BAG_28', 'AIR_BAG_97', 'AIR_BAG_98', 'AIR_BAG_99', 'EJECTION_0', 'EJECTION_1', 'EJECTION_2', 'EJECTION_3', 'EJECTION_7', 'EJECTION_8', 'EJECTION_9', 'EJ_PATH_0', 'EJ_PATH_1', 'EJ_PATH_2', 'EJ_PATH_3', 'EJ_PATH_4', 'EJ_PATH_5', 'EJ_PATH_6', 'EJ_PATH_7', 'EJ_PATH_8', 'EJ_PATH_9', 'P_SF1_0', 'P_SF1_5', 'P_SF1_8', 'P_SF1_9', 'P_SF1_13', 'P_SF1_18', 'P_SF1_21', 'P_SF1_32', 'P_SF1_37', 'P_SF1_42', 'P_SF1_56', 'P_SF1_60', 'P_SF1_61', 'P_SF1_62', 'P_SF1_64', 'P_SF1_65', 'P_SF1_66', 'P_SF1_68', 'P_SF1_72', 'P_SF1_75', 'P_SF1_76', 'P_SF1_80', 'P_SF1_82', 'P_SF1_86', 'P_SF1_87', 'P_SF1_88', 'P_SF1_89', 'P_SF1_90', 'P_SF1_91', 'P_SF1_92', 'P_SF1_99', 'P_SF2_0', 'P_SF2_5', 'P_SF2_66', 'P_SF2_82', 'P_SF2_99', 'P_SF3_0', 'P_SF3_99', 'VEH_NO', 'PER_NO', 'STR_VEH', 'TOW_VEH', 'EMER_USE', 'ROLLOVER', 'FIRE_EXP', 'INJ_SEV', 'ALC_RES']


# We should have a look at the max and means by ST_CASE. It will depend on the feature if we want max or mean for that feature.
persons_max_by_ST_CASE = max_by_group(dummied_persons, 'ST_CASE')
persons_min_by_ST_CASE = min_by_group(dummied_persons, 'ST_CASE')
persons_mean_by_ST_CASE = mean_by_group(dummied_persons, 'ST_CASE')


# Cool! Let's think about where we want the max, and where we want the mean, feature by feature and collect them to prepare to merge into accidents. In some cases, the mean and the max should be the same thing.
persons_max_approved = persons_max_by_ST_CASE[all_cols_for_max]
persons_min_approved = persons_min_by_ST_CASE[['BODY_TYP']]
persons_mean_approved = persons_mean_by_ST_CASE[['MOD_YEAR', 'EXTRICAT', 'DRINKING', 'DRUGS', 'WORK_INJ', 'AGE']]


# Now we'll finally merge everything onto accidents
df = accidents.merge(persons_max_approved, right_index=True, left_index=True)
df = df.merge(persons_min_approved, right_index=True, left_index=True)
df = df.merge(persons_mean_approved, right_index=True, left_index=True)
