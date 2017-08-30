import pandas as pd
from NJ_clean_factorize import NJ
from capstone_salad import max_by_group, mean_by_group



def load_process_occupants_data(filename='/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Occupants/occupants_subsample.csv'):

    NJ_occupants = pd.read_csv(filename, error_bad_lines=False, dtype='object')

    my_cols = {'1':'case_num', '3':'vehicle_number', '3':'occupant_number', '4':'physical_condition', \
                '5':'position_in_on_vehicle', '6':'ejection_code', '7':'age', '8':'sex', '9':'location_of_most_severe_injury', \
                '11':'refused_medical_attention', '12':'safety_equiptment_available', '13':'safety_equiptment_used', \
                '14':'airbag_deployment', '15':'hospital_code'}
    #NJ_occupants.drop('Unnamed: 0', axis=1, inplace=True)
    #NJ_occupants.rename(columns=my_cols, inplace=True)

    return NJ_occupants

def engineer_features_NJ_occupants():
    NJ_occupants['ejection_bool'] = NJ_occupants['6'].isin(['03', '04', '02'])
    NJ_occupants['7'] = pd.to_numeric(NJ_occupants['7'], errors='coerce')
    NJ_occupants['minors_involved'] = NJ_occupants['7'] <= 18
    NJ_occupants['elderly_involved'] = NJ_occupants['7'] >= 62
    return NJ_occupants

def prep_occupants_and_merge():
    agg = NJ_occupants.groupby('1')
    occupants_maxed_group = agg[['minors_involved', 'elderly_involved', 'ejection_bool']].max()
    return pd.merge(NJ, occupants_maxed_group, left_on='0', right_index=True, how='left')



if __name__ == '__main__':
    NJ_occupants = load_process_occupants_data()
    engineer_features_NJ_occupants()
    NJ = prep_occupants_and_merge()
