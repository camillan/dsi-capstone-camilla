from NJ_occupants_clean import NJ
from capstone_salad import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

NJ_for_model = NJ[[u'40', '17', u'29', 'involves_injury', u'5', '41', 'ejection_bool', '26', '30']].apply(pd.to_numeric, errors='coerce')

# features used:
# 5: time of day
# 18: total vehicles involved
# 17: type of road
# 26: road character
# 29: light condition
# 40: posted speed
# 41: posted speed at cross street
# I may want to add cell phone and speed information

NJ_for_model.dropna(inplace=True)
NJ_for_model.reset_index(level=0, drop=True, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values, test_size=0.3)

my_forest = RandomForestClassifier(n_jobs=4, max_depth=5)
my_forest.fit(X_train, y_train)
#my_forest.predict_proba(X_train)
print my_forest.score(X_test, y_test)
