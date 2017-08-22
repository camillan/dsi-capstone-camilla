import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def rf_model_pipeline(X, y):

    '''
    Takes in X variables (in a list) and the name of the y column and builds a Random Forest Classifier.
    Returns the model itself as an object, and prints the top features in order of importance for further investigation.
    '''

    my_forest = RandomForestClassifier()
    my_forest.fit(X, y)
    my_forest.predict_proba(X)
    feature_importances = np.argsort(my_forest.feature_importances_)
    print "top features:", list(X[feature_importances[-1::-1]])

    #print my_forest.score(X_test, y_test)

    return my_forest



if __name__ == '__main__':

    vehicles = pd.read_csv('FARS2015NationalCSV/vehicle.csv')

    X_train, X_test, y_train, y_test = train_test_split(vehicles.drop('DEFORMED', axis=1), vehicles.DEFORMED, test_size=0.3)

    # For the first model:
    X = X_train[['NUMOCCS', 'BODY_TYP', 'HAZ_INV', 'TRAV_SP', 'ROLLOVER', 'FIRE_EXP', 'DEATHS', 'PREV_ACC', 'SPEEDREL', 'VNUM_LAN', 'VSURCOND', 'VPROFILE', 'VPAVETYP', 'P_CRASH3', 'ACC_TYPE']]

    rf_model_pipeline(X, y_train)
