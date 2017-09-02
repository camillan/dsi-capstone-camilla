from capstone_salad import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import csv
from NJ_load_preprocess import *

def rf_11(df):
    NJ_for_model = df[['involves_injury',\
    'ejection_bool', \
    'same_direction_crash', \
    'opposite_direction_crash', \
    'right_angle_crash', \
    'overturn', \
    'backing_up', \
    'left_or_u_turn', \
    'pedacyclist', \
    'pedestrian', \
    'railcar', \
    'bus_or_truck', \
    'motorcycles', \
    'snow_or_icy', \
    'dark_no_street_lights', \
    'rain', \
    '18', \
    '5', \
    '40', \
    'unsafe_speed', \
    'wrong_way', \
    'teen_driver']].apply(pd.to_numeric, errors='coerce')

    # available features from base table:
    # 5: time of day
    # 18: total vehicles involved
    # 17: type of road
    # 26: road character
    # 29: light condition
    # 40: posted speed
    # 41: posted speed at cross street


    # initial train and test split
    NJ_for_model.dropna(inplace=True)
    NJ_for_model.reset_index(level=0, drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values, test_size=0.3)

    # k folds for validation
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)

    scores = []

    for train_indices, test_indices in kf.split(X_train):
        my_forest = RandomForestClassifier(n_jobs=4, max_depth=10, n_estimators=35)
        my_forest.fit(X_train[train_indices], y_train[train_indices])
        scores.append(my_forest.score(X_train[test_indices], y_train[test_indices]))
        #my_forest.predict_proba(X_train)

    print np.mean(scores)
    feature_importances = np.argsort(my_forest.feature_importances_)
    print "top features:", list(NJ_for_model.drop('involves_injury', axis=1).columns[feature_importances[::-1]])
    return my_forest



def main():
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
    return NJ


if __name__ == '__main__':
    NJ = main()
    # Run the model
    rf_11_object = rf_11(NJ)