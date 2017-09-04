
from sklearn.grid_search import GridSearchCV

from NJ_rf11 import *

NJ = load_NJ()
NJ = preprocess_NJ_accidents_cols(NJ)

print 'done w/ accidents load'

# Occupants table
NJ_occupants = load_process_occupants_data()
NJ_occupants = engineer_features_NJ_occupants(NJ_occupants)
NJ = prep_occupants_and_merge(NJ_occupants, NJ)

print 'done w/ occupants'

# Vehicles table
my_list = NJ['0']
NJ_vehicles = load_process_vehicles_data(my_list=my_list)
NJ_vehicles = engineer_features_NJ_vehicles(NJ_vehicles)
NJ = prep_vehicles_and_merge(NJ_vehicles, NJ)


print 'done w/ vehicles'

my_forest = rf_11(NJ)


NJ_for_model = NJ[['involves_injury',\
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
    'snow_or_icy', \
    'rain', \
    '18', \
    '40', \
    'unsafe_speed', \
    'teen_driver', \
    'num_ppl_involved']].apply(pd.to_numeric, errors='coerce')

NJ_for_model.dropna(inplace=True)
NJ_for_model.reset_index(level=0, drop=True, inplace=True)

print 'done w/ model fitting, now time for grid searching'

param_grid = { "n_estimators"      : [10, 20, 30],\
           "max_features"      : [5, 8, 12],\
           "max_depth"         : [10, 20, None],\
           "min_samples_split" : [2, 4]}

grid_searched_rf = GridSearchCV(estimator=my_forest, param_grid=param_grid, n_jobs=-1)
grid_searched_rf.fit(NJ_for_model.drop('involves_injury', axis=1).values, NJ_for_model['involves_injury'].values)

print grid_searched_rf.best_estimator_


'''RESULTS
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features=12, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=30, n_jobs=4, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

            '''
