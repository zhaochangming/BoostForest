from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from .BT import BoostTreeClassifier, BoostTreeRegressor


class BoostForestClassifier(BaggingClassifier):
    def __init__(self, max_leafs=5, node_model='ridge_clip', min_sample_leaf_list=None, reg_alpha_list=None, max_depth=None,
                 elm_hidden_layer_nodes=None, **kwargs):
        BaggingClassifier.__init__(self,
                                   base_estimator=BoostTreeClassifier(max_leafs=max_leafs, node_model=node_model,
                                                                      min_sample_leaf_list=min_sample_leaf_list,
                                                                      reg_alpha_list=reg_alpha_list, max_depth=max_depth,
                                                                      elm_hidden_layer_nodes=elm_hidden_layer_nodes), **kwargs)


class BoostForestRegressor(BaggingRegressor):
    def __init__(self, max_leafs=5, node_model='ridge_clip', min_sample_leaf_list=None, reg_alpha_list=None, max_depth=None,
                 elm_hidden_layer_nodes=None, **kwargs):
        BaggingRegressor.__init__(self,
                                  base_estimator=BoostTreeRegressor(max_leafs=max_leafs, node_model=node_model,
                                                                    min_sample_leaf_list=min_sample_leaf_list,
                                                                    reg_alpha_list=reg_alpha_list, max_depth=max_depth,
                                                                    elm_hidden_layer_nodes=elm_hidden_layer_nodes), **kwargs)
