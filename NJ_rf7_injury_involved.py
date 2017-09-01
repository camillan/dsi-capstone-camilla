from capstone_salad import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import csv
from NJ_load_preprocess import *
from NJ_occupants_load_preprocess import *

def rf_7(df):
    NJ_for_model = df[[u'5', '17', '18', '26', u'29', '30', u'40', '41', 'involves_injury', 'ejection_bool']].apply(pd.to_numeric, errors='coerce')

    # features used:
    # 5: time of day
    # 18: total vehicles involved
    # 17: type of road
    # 26: road character
    # 29: light condition
    # 30: environmetal condition
    # 40: posted speed
    # 41: posted speed at cross street

    # initial train and test split
    NJ_for_model.dropna(inplace=True)
    NJ_for_model.reset_index(level=0, drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values, test_size=0.3)

    # k folds for validation
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)

    for train_indices, test_indices in kf.split(X_train):
        my_forest = RandomForestClassifier(n_jobs=4, max_depth=10)
        my_forest.fit(X_train[train_indices], y_train[train_indices])
        #my_forest.predict_proba(X_train)
        print my_forest.score(X_train[test_indices], y_train[test_indices])
        feature_importances = np.argsort(my_forest.feature_importances_)
        print "12: top five:", list(df.columns[feature_importances[-1:-6:-1]])
    # gives you a score of 0.769727604605

if __name__ == '__main__':
    NJ = load_NJ()
    NJ = preprocess_cols(NJ)
    NJ_occupants = load_process_occupants_data()
    engineer_features_NJ_occupants(NJ_occupants)
    NJ = prep_occupants_and_merge(NJ_occupants, NJ)
    rf_7(NJ)
