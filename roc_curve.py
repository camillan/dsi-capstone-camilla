from sklearn.metrics import roc_curve
from capstone_salad import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import csv
from NJ_load_preprocess import *


def ROC_plot_on_final_model(df):
    '''
    Playing with the model for ROC curve.
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
    X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values, test_size=0.3)

    # k folds for validation
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)

    scores = []

    # for ROC plot
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    for train_indices, test_indices in kf.split(X_train):
        my_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=10, max_features=5, max_leaf_nodes=None,
                        min_impurity_split=1e-07, min_samples_leaf=1,
                        min_samples_split=4, min_weight_fraction_leaf=0.0,
                        n_estimators=30, n_jobs=4, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
        my_forest.fit(X_train[train_indices], y_train[train_indices])
        scores.append(my_forest.score(X_train[test_indices], y_train[test_indices]))
        fpr, tpr, _ = roc_curve(y_train[test_indices], my_forest.predict_proba(X_train)[test_indices][:,1])

        plt.plot(fpr, tpr, label='ROC fold %d' % i)
        i += 1

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

    #print np.mean(scores)
    #feature_importances = np.argsort(my_forest.feature_importances_)
    #print "top features:", list(NJ_for_model.drop('involves_injury', axis=1).columns[feature_importances[::-1]])
    #return my_forest



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

    # Run the plot
    ROC_plot_on_final_model(NJ)
