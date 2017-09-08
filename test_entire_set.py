from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import csv
from load_whole_NJ_dataset import *

def whole_NJ_dataset_rf_12(df):
    '''
    Takes in the master matrix (over 4 million rows) and puts it into the most final version of the model.
    Note: a couple of features are taken out to solve NAN erros and improve processing time.
    '''

    NJ_for_model = df[['involves_injury',\
    'no_median_barrier', \
    'same_direction_crash', \
    'opposite_direction_crash', \
    'right_angle_crash', \
    'overturn', \
    'backing_up', \
    'left_or_u_turn', \
    'pedacyclist', \
    'pedestrian', \
    'snow_or_icy', \
    'rain', \
    '18', \
    '40']].apply(pd.to_numeric, errors='coerce')

    # Looking at correlations too will help us understand feature importances, and make certain that the data is lining up correctly.
    print NJ_for_model.corr()['involves_injury']

    # available features from base table:
    # 5: time of day
    # 18: total vehicles involved
    # 17: type of road
    # 26: road character
    # 29: light condition
    # 40: posted speed
    # 41: posted speed at cross street

    # took out: railcar, wrong way, bus or truck, dark_no_street_lights, 5 (time) and it improved the model


    # initial train and test split
    NJ_for_model.dropna(inplace=True)
    NJ_for_model.reset_index(level=0, drop=True, inplace=True)
    print NJ_for_model.shape
    X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values, test_size=0.3)

    # k folds for validation
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)

    scores = []

    for train_indices, test_indices in kf.split(X_train):
        my_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=10, max_features=5, max_leaf_nodes=None,
                        min_impurity_split=1e-07, min_samples_leaf=1,
                        min_samples_split=4, min_weight_fraction_leaf=0.0,
                        n_estimators=30, n_jobs=4, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
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
    NJ_vehicles = load_process_vehicles_data()
    NJ_vehicles = engineer_features_NJ_vehicles(NJ_vehicles)
    NJ = prep_vehicles_and_merge(NJ_vehicles, NJ)
    return NJ


if __name__ == '__main__':
    NJ = main()
    # Run the model
    rf_12_object = whole_NJ_dataset_rf_12(NJ)

'''
Results below, so you don't have to run it each time.


In [7]: run NJ_rf12_whole_dataset.py
involves_injury             1.000000
no_median_barrier          -0.025812
same_direction_crash       -0.002544
opposite_direction_crash    0.025028
right_angle_crash           0.044805
overturn                    0.041656
backing_up                 -0.068977
left_or_u_turn              0.025298
pedacyclist                 0.062788
pedestrian                  0.111092
snow_or_icy                -0.026933
rain                        0.014090
18                          0.010609
40                          0.100141
Name: involves_injury, dtype: float64
(4401722, 14)
0.772070342609
top features: ['18', '40', 'pedestrian', 'pedacyclist', 'right_angle_crash', 'backing_up', 'overturn', 'snow_or_icy', 'no_median_barrier', 'same_direction_crash', 'opposite_direction_crash', 'left_or_u_turn', 'rain']

Second time

involves_injury             1.000000
no_median_barrier          -0.025812
same_direction_crash       -0.002544
opposite_direction_crash    0.025028
right_angle_crash           0.044805
overturn                    0.041656
backing_up                 -0.068977
left_or_u_turn              0.025298
pedacyclist                 0.062788
pedestrian                  0.111092
snow_or_icy                -0.026933
rain                        0.014090
18                          0.010609
40                          0.100141
Name: involves_injury, dtype: float64
(4401722, 14)
0.772235862268
top features: ['18', '40', 'pedestrian', 'pedacyclist', 'right_angle_crash', 'backing_up', 'overturn', 'snow_or_icy', 'no_median_barrier', 'same_direction_crash', 'opposite_direction_crash', 'left_or_u_turn', 'rain']
'''
