Python-package Introduction
===========================

This document gives a basic walk-through of BoostForest Python-package.

**List of other helpful links**



-  `Python API <BoostForest.html>`__

-  `Parameters Tuning <Parameters-Tuning.html>`__

-  `Python Examples <Python-Examples.html>`__

Install (in the future)
-------

The preferred way to install BoostForest is via pip from `Pypi <https://pypi.org/project/BoostForest>`__:

::

    pip install BoostForest


To verify your installation, try to ``import BoostForest`` in Python:

::

    import BoostForest

Data Interface
--------------

The BoostForest Python module can load data from:

-  NumPy 2D array(s)
    .. code:: python

        import numpy as np
        data = np.random.rand(500, 10)
        label = np.random.randint(2, size=500)



Setting Parameters
------------------

BoostForest can use a dictionary to set parameters.
For instance:

-  When min_sample_leaf_list and reg_alpha_list are certain values:

   .. code:: python

       param = {'max_leafs': 5, 'node_model': 'Ridge', 'min_sample_leaf_list':5, 'reg_alpha_list': 0.1, 'max_depth': None, 'elm_hidden_layer_nodes': 100, 'random_state':0}

-  When min_sample_leaf_list and reg_alpha_list are lists:

   .. code:: python

       param = {'max_leafs': 5, 'node_model': 'Ridge', 'min_sample_leaf_list': [5, 6, 7], 'reg_alpha_list': [0.1, 0.5, 1.0], 'max_depth': None, 'elm_hidden_layer_nodes': 100, 'random_state':0}


Training
--------

Training a model requires a parameter dictionary and data set:

.. code:: python


    estimator = BoostForest.BoostTreeClassifier(**param).fit(data, label)

After training, the model can be saved:

.. code:: python

    estimator.save_model('model.joblib')

A saved model can be loaded:

.. code:: python

    import joblib
    estimator = joblib.load('model.joblib')


Predicting
----------

A model that has been trained or loaded can perform predictions on datasets:

.. code:: python

    # 7 entities, each contains 10 features
    data = np.random.rand(7, 10)
    ypred = estimator.predict(data)
