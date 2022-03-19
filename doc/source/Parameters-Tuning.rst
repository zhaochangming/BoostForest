Parameters Tuning
=================

This page contains parameters tuning guides for different scenarios.

For Better Accuracy
-------------------

-  Use large ``n_nodes_H`` and ``n_groups_Z`` (may be slower)

-  Use bigger training data


Deal with Over-fitting
----------------------

-  Use more ``k_neighbors`` for semi-supervised learning

-  Use bigger training data

-  Try ``reg_alpha``, ``reg_lambda``, ``reg_laplacian`` and ``sigma`` for regularization

-  Try ``n_nodes_Z`` for generating suitable feature nodes


