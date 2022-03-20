from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from .BT import BoostTreeClassifier, BoostTreeRegressor


class BoostForestClassifier(BaggingClassifier):
    """
        Construct a BoostForestClassifier.

        Parameters
        ----------
        max_leafs : int, optional (default=5)
            Maximum tree leaves for BoostTree.
        node_model : str, ['Ridge', 'ELM']
            Controls the node model.
        min_sample_leaf_list : int or list (default=1)
            Controls the minimum number of data needed in a leaf.
        reg_alpha_list : float, optional (default=0.1)
            L2 regularization term on weights.
        max_depth : int, optional (default=None)
            Maximum tree depth for BoostTree, None means no limit.
        elm_hidden_layer_nodes : int or list (default=100)
            Controls the number of ELM's hidden layer nodes, when using ELM as the node model.
        random_state : int, default=0
                        Controls the randomness of the estimator.

        n_estimators : int, default=10
            The number of base estimators in the ensemble.

        max_samples : int or float, default=1.0
            The number of samples to draw from X to train each base estimator (with
            replacement by default, see `bootstrap` for more details).

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator (
            without replacement by default, see `bootstrap_features` for more
            details).

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

        bootstrap : bool, default=True
            Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.

        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate
            the generalization error.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble. See :term:`the Glossary <warm_start>`.

        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`.

        random_state : int or RandomState, default=None
            Controls the random resampling of the original dataset
            (sample wise and feature wise).
            If the base estimator accepts a `random_state` attribute, a different
            seed is generated for each instance in the ensemble.
            Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, max_leafs=5, node_model='ridge_clip', min_sample_leaf_list=None, reg_alpha_list=None, max_depth=None,
                 elm_hidden_layer_nodes=None, **kwargs):
        BaggingClassifier.__init__(self,
                                   base_estimator=BoostTreeClassifier(max_leafs=max_leafs, node_model=node_model,
                                                                      min_sample_leaf_list=min_sample_leaf_list,
                                                                      reg_alpha_list=reg_alpha_list, max_depth=max_depth,
                                                                      elm_hidden_layer_nodes=elm_hidden_layer_nodes), **kwargs)


class BoostForestRegressor(BaggingRegressor):
    """
        Construct a BoostForestClassifier.

        Parameters
        ----------
        max_leafs : int, optional (default=5)
            Maximum tree leaves for BoostTree.
        node_model : str, ['Ridge', 'ELM']
            Controls the node model.
        min_sample_leaf_list : int or list (default=1)
            Controls the minimum number of data needed in a leaf.
        reg_alpha_list : float, optional (default=0.1)
            L2 regularization term on weights.
        max_depth : int, optional (default=None)
            Maximum tree depth for BoostTree, None means no limit.
        elm_hidden_layer_nodes : int or list (default=100)
            Controls the number of ELM's hidden layer nodes, when using ELM as the node model.
        random_state : int, default=0
                        Controls the randomness of the estimator.
    """
    def __init__(self, max_leafs=5, node_model='ridge_clip', min_sample_leaf_list=None, reg_alpha_list=None, max_depth=None,
                 elm_hidden_layer_nodes=None, **kwargs):
        BaggingRegressor.__init__(self,
                                  base_estimator=BoostTreeRegressor(max_leafs=max_leafs, node_model=node_model,
                                                                    min_sample_leaf_list=min_sample_leaf_list,
                                                                    reg_alpha_list=reg_alpha_list, max_depth=max_depth,
                                                                    elm_hidden_layer_nodes=elm_hidden_layer_nodes), **kwargs)
