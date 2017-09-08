from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from capstone_salad import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import csv
from NJ_load_preprocess import *

def adaboost_model(df):
    '''
    This takes the final model's features and puts them into an AdaBoostClassifier to compare the output to the
    Random Forest classifier.
    '''

    NJ_for_model = df[['involves_injury',\
    'ejection_bool', \
    'no_median_barrier', \
    'same_direction_crash', \
    'opposite_direction_crash', \
    'right_angle_crash', \
    'overturn', \
    'backing_up', \
    'left_or_u_turn', \
    'pedacyclist', \
    'pedestrian', \
    'motorcycles', \
    'rain', \
    'snow_or_icy',\
    '5',\
    '18', \
    '40', \
    'unsafe_speed', \
    'teen_driver', \
    'elderly_involved',\
    'minors_involved',\
    'num_ppl_involved']].apply(pd.to_numeric, errors='coerce')

    print NJ_for_model.corr()['involves_injury']

    # available features from base table:
    # 5: time of day
    # 18: total vehicles involved
    # 17: type of road
    # 26: road character
    # 29: light condition
    # 40: posted speed
    # 41: posted speed at cross street

    # took out: railcar, wrong way, bus or truck, dark_no_street_lights, 5 (time), snow or icy and it improved the model


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
        my_adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
        my_adaboost.fit(X_train[train_indices], y_train[train_indices])
        scores.append(my_adaboost.score(X_train[test_indices], y_train[test_indices]))
        #my_forest.predict_proba(X_train)

    print np.mean(scores)
    feature_importances = np.argsort(my_adaboost.feature_importances_)
    print "top features:", list(NJ_for_model.drop('involves_injury', axis=1).columns[feature_importances[::-1]])
    return my_adaboost



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
    adaboost_object_1 = adaboost_model(NJ)


'''
Results:

0.782280766929
top features: ['pedacyclist', 'pedestrian', 'ejection_bool', 'overturn', '18', 'backing_up', 'opposite_direction_crash', 'num_ppl_involved', 'left_or_u_turn', 'right_angle_crash', '40', 'snow_or_icy', 'rain', 'minors_involved', '5', 'unsafe_speed', 'teen_driver', 'elderly_involved', 'same_direction_crash', 'no_median_barrier', 'motorcycles']

Interesting that the feature importances are a noticable bit different than when using random forest.

'''
