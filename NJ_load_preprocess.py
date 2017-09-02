import pandas as pd
import csv


def load_NJ(filename='/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14/NJ_master_subsample.csv'):
    NJ = pd.read_csv(filename, quoting=csv.QUOTE_NONE)
    return NJ


def preprocess_NJ_accidents_cols(df):
    '''
    Factorize columns according to the accidents sample csv and the accidents manual and list of columns.
    '''
    df[u'13'] = pd.factorize(df[u'13'])[0] # severity - don't include in models because of leaky
    df[u'4'] = pd.factorize(df[u'4'])[0] # day of week
    df['44'] = pd.factorize(df['44'])[0] # cell phone use
    df['30'] = pd.to_numeric(df['30'], errors='coerce')
    df['snow_or_icy'] = df['30'].isin([3, 6, 7, 8])
    df['rain'] = df['30'] == 2
    df['hilly_road'] = df['27'].isin(['05', '06', '07', '08'])
    df['same_direction_crash'] = df['17'].isin(['01', '02'])
    df['dark_no_street_lights'] = df['29'].isin(['04', '05'])
    df['right_angle_crash'] = df['17'] == '03'
    df['opposite_direction_crash'] = df['17'].isin(['04', '05'])
    df['overturn'] = df['17'] == '10'
    df['backing_up'] = df['17'] == '08'
    df['left_or_u_turn'] = df['17'] == '07'
    df['pedacyclist'] = df['17'] == '14'
    df['pedestrian'] = df['17'] == '13'
    df['railcar'] = df['17'] == '16'
    df['involves_injury'] = df[u'13'].isin([1,2]) # the predicting problem!
    return df


def load_process_occupants_data(filename='/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Occupants/occupants_subsample.csv'):

    NJ_occupants = pd.read_csv(filename, error_bad_lines=False, dtype='object')

    my_cols = {'1':'case_num', '3':'vehicle_number', '3':'occupant_number', '4':'physical_condition', \
                '5':'position_in_on_vehicle', '6':'ejection_code', '7':'age', '8':'sex', '9':'location_of_most_severe_injury', \
                '11':'refused_medical_attention', '12':'safety_equiptment_available', '13':'safety_equiptment_used', \
                '14':'airbag_deployment', '15':'hospital_code'}
    #NJ_occupants.drop('Unnamed: 0', axis=1, inplace=True)
    #NJ_occupants.rename(columns=my_cols, inplace=True)

    return NJ_occupants


def engineer_features_NJ_occupants(df):
    df['ejection_bool'] = df['6'].isin(['03', '04', '02'])
    df['7'] = pd.to_numeric(df['7'], errors='coerce')
    df['3'] = pd.to_numeric(df['3'], errors='coerce')
    df['minors_involved'] = df['7'] <= 18
    df['elderly_involved'] = df['7'] >= 62
    df['teen_driver'] = df['7'].isin(range(15,19)) & df['3'] == 1
    return df


def prep_occupants_and_merge(df, other_df):
    '''
    Creates an aggregate object and gets the features we want from it for the master table.
    Merges that info into the master table, NJ.
    '''
    agg = df.groupby('1')
    occupants_maxed_group = agg[['minors_involved', 'elderly_involved', 'ejection_bool', 'teen_driver']].max()
    return pd.merge(other_df, occupants_maxed_group, left_on='0', right_index=True, how='left')


def load_process_vehicles_data(my_list, filename='/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Vehicles/NJ_vehicles_subsample.csv'):
    '''
    Iterate over a large csv and return the lines where the unique identifier matches the unique values in my_list.
    For use when opening the vehicles .csv file - but only the ones where it's a crash that's also in the subsample of NJ crash data (the accidents file).

    iter_csv = pd.read_csv(filename, iterator=True, chunksize=1000, header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    NJ_vehicles = pd.concat([chunk[chunk[1].isin(my_list)] for chunk in iter_csv])
    '''
    NJ_vehicles = pd.read_csv(filename, header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    return NJ_vehicles


def engineer_features_NJ_vehicles(df):
    '''
    Engineer/extract features as interpreted from the crash manual.
    '''
    df[16] = pd.to_numeric(df[16], errors='coerce')
    df['bus_or_truck'] = df[16] > 19 # vehicle type
    df['suv'] = df[16] == 04
    df['passenger_or_cargo_van'] = df[16].isin([2, 3])
    df['passenger_vehicle'] = df[16] == 1
    df['motorcycles'] = df[16] == 8
    df[23] = pd.to_numeric(df[23], errors='coerce')
    df['improper_turning'] = df[23] == 8
    df['wrong_way'] = df[23] == 12
    df['unsafe_speed'] = df[23] == 1
    return df


def prep_vehicles_and_merge(df, other_df):
    '''
    Creates an aggregate object and gets the features we want from it for the master table.
    Merges that info into the master table, NJ.
    '''
    agg = df.groupby(2)
    vehicles_maxed_group = agg[['bus_or_truck', 'suv', 'passenger_or_cargo_van', 'passenger_vehicle', 'motorcycles', 'improper_turning', 'wrong_way', 'unsafe_speed']].max()
    return pd.merge(other_df, vehicles_maxed_group, left_on='0', right_index=True, how='left')





if __name__ == '__main__':

    # Accidents table
    NJ = load_NJ()
    NJ = preprocess_NJ_accidents_cols(NJ)

    # Occupants table
    NJ_occupants = load_process_occupants_data()
    NJ_occupants = engineer_features_NJ_occupants(NJ_occupants)
    NJ = prep_occupants_and_merge(NJ_occupants, NJ)

    # Vehicles table
    my_list = NJ['0']
    NJ_vehicles = load_process_vehicles_data(my_list=my_list)
    NJ_vehicles = engineer_features_NJ_vehicles(NJ_vehicles)
    NJ = prep_vehicles_and_merge(NJ_vehicles, NJ)
