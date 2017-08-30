import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def rf_model_pipeline(X_train, X_test, y_train, y_test):
    '''
    Takes in training and test data and builds a Random Forest Classifier.
    Returns the model itself as an object, and prints the top features in order of importance for further investigation.
    '''
    my_forest = RandomForestClassifier(n_jobs=3)
    my_forest.fit(X_train, y_train)
    my_forest.predict_proba(X_train)
    print my_forest.score(X_test, y_test)
    return my_forest


def check_nan_numeric_shape_change(df):
    '''
    Shows how much the shape of a DF changes based on dropping NANs and object columns.
    '''
    df_shape = df.shape
    df_cut = drop_object_cols_and_NANs(df).shape
    return [df_shape, df_cut]


def drop_object_cols_and_NANs(df):
    '''Takes a dataframe and drops columns that are objects and rows that contain missing values'''
    df.dropna(inplace=True)
    df = df.select_dtypes(exclude=['object'])
    return df


def mean_by_group(dataframe, col):
    '''
    Groups the data by a column and returns the mean per group.
    '''
    return dataframe.groupby(col).mean()


def max_by_group(dataframe, col):
    '''
    Groups the data by a column and returns the max per group.
    '''
    return dataframe.groupby(col).max()


def min_by_group(dataframe, col):
    '''
    Groups the data by a column and returns the min per group.
    '''
    return dataframe.groupby(col).min()


def cols_not_shared(df1, df2):
    '''
    Print a list of columns that are in the second DF without being in the first.
    For merge purposes.
    '''
    return [col for col in df2.columns if col not in df1.columns]
