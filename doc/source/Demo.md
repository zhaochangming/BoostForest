# BoostForest
This page shows some Python demos.
``ConcreteFlow.joblib``, ``Sonar.joblib`` and ``Seeds.joblib`` can be downloaded at [data](https://github.com/zhaochangming/BoostForest/tree/main/data).
## BoostForest for Binary Classification


```python
import joblib
import numpy as np
from BoostForest import BoostTreeClassifier, BoostForestClassifier
from sklearn.ensemble import RandomForestClassifier

data = joblib.load('data/Sonar.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = RandomForestClassifier(n_estimators=50)
model.fit(training_X, training_y)
print('ACC of Baseline: %.3f' % model.score(testing_X, testing_y))
model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('ACC of BoostTree: %.3f'% model.score(testing_X, testing_y))
model = BoostForestClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1, n_estimators=50)
model.fit(training_X, training_y)
print('ACC of BoostForest: %.3f' % model.score(testing_X, testing_y))

```

    ACC of Baseline: 0.929
    ACC of BoostTree: 0.833
    ACC of BoostForest: 0.905


## BoostForest for Multi-class Classification


```python
import joblib
import numpy as np
from BoostForest import BoostTreeClassifier,  BoostForestClassifier
from sklearn.ensemble import RandomForestClassifier
data = joblib.load('data/Seeds.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = RandomForestClassifier(n_estimators=50)
model.fit(training_X, training_y)
print('ACC of Baseline: %.3f' % model.score(testing_X, testing_y))
model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('ACC of BoostTree: %.3f'% model.score(testing_X, testing_y))
model = BoostForestClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1, n_estimators=50)
model.fit(training_X, training_y)
print('ACC of BoostForest: %.3f' % model.score(testing_X, testing_y))
```

    ACC of Baseline: 0.881
    ACC of BoostTree: 0.881
    ACC of BoostForest: 0.929


## BoostForest for Regression

```python
import joblib
import numpy as np
from BoostForest import BoostTreeRegressor, BoostForestRegressor
from sklearn.ensemble import  RandomForestRegressor
data = joblib.load('data/ConcreteFlow.joblib')
training_X, training_y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
testing_X, testing_y = data['testing_data'], data['testing_label']
model = RandomForestRegressor(n_estimators=50)
model.fit(training_X, training_y)
print('R^2 of Baseline: %.3f' % model.score(testing_X, testing_y))
model = BoostTreeRegressor(max_leafs=None, min_sample_leaf_list=2, reg_alpha_list=0.1)
model.fit(training_X, training_y)
print('R^2 of BoostTree: %.3f'% model.score(testing_X, testing_y))
model = BoostForestRegressor(max_leafs=None, min_sample_leaf_list=2, reg_alpha_list=0.1, n_estimators=50)
model.fit(training_X, training_y)
print('R^2 of BoostForest: %.3f' % model.score(testing_X, testing_y))
```

    R^2 of Baseline: 0.555
    R^2 of BoostTree: 0.558
    R^2 of BoostForest: 0.591

