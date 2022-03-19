import joblib
import numpy as np
from BoostForest import BoostTreeClassifier, BoostTreeRegressor, BoostForestRegressor, BoostForestClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor



data = joblib.load('data/Sonar.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('BoostTree: ',model.score(testing_X, testing_y))
model = BoostForestClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1,n_estimators=50)
model.fit(training_X, training_y)
print('BoostForest: ',model.score(testing_X, testing_y))
model = RandomForestClassifier(n_estimators=50)
model.fit(training_X, training_y)
print('RandomForest: ',model.score(testing_X, testing_y))

data = joblib.load('data/Seeds.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('BoostTree: ',model.score(testing_X, testing_y))
model = BoostForestClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1,n_estimators=50)
model.fit(training_X, training_y)
print('BoostForest: ',model.score(testing_X, testing_y))
model = RandomForestClassifier(n_estimators=50)
model.fit(training_X, training_y)
print('RandomForest: ',model.score(testing_X, testing_y))

data = joblib.load('data/ConcreteFlow.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = BoostTreeRegressor(max_leafs=None, min_sample_leaf_list=2, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('BoostTree: ',model.score(testing_X, testing_y))
model = BoostForestRegressor(max_leafs=None, min_sample_leaf_list=2, reg_alpha_list=0.1,n_estimators=50)
model.fit(training_X, training_y)
print('BoostForest: ',model.score(testing_X, testing_y))
model = RandomForestRegressor(n_estimators=50)
model.fit(training_X, training_y)
print('RandomForest: ',model.score(testing_X, testing_y))