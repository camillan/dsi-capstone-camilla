from capstone_salad import *
import pandas as pd
import csv

NJ = pd.read_csv('/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14/NJ_master.csv', quoting=csv.QUOTE_NONE)

#Notes:
#Column 9: lines up with total killed.
#10: total injured
#11: pedestrians killed
#12: pedestrians injured
#13: severity
# any of these could be the labels

# Factorize the
NJ[u'13'] = pd.factorize(NJ[u'13'])[0]



#NJ_for_model = NJ.loc[['year', 'county', 'municipality', 'month', '5', '12', '13', '15', '16', '18']].apply(pd.to_numeric, errors='coerce')
NJ_for_model = NJ[[u'month', u'county', u'18', u'26', u'29', u'13']].apply(pd.to_numeric, errors='coerce')
# features used:
# 18: total vehicles involved
# 26: road character
# 29: light condition
# I may want to add cell phone and speed information



NJ_for_model.dropna(inplace=True)
NJ_for_model.reset_index(level=0, drop=True, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(NJ_for_model.drop(u'13', axis=1).values, NJ_for_model[u'13'].values, test_size=0.6)

my_forest = rf_model_pipeline(X_train, X_test, y_train, y_test)
# This gives you an accuracy of 0.764000876694
