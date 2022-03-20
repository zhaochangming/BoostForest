import numpy as np
from copy import deepcopy
from sklearn.metrics import log_loss, mean_squared_error
from scipy.special import softmax, expit
import random
from sklearn.linear_model import Ridge
import sys
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import joblib

sys.setrecursionlimit(100000000)
_MACHINE_EPSILON = np.finfo(np.float64).eps
max_response = 4
min_sample_leaf_list = range(5, 16)


class ridge_clip:
    def __init__(self, alpha=None, coef=None, bias=None):
        self.model = Ridge(alpha=alpha)
        self.alpha = alpha
        self.upper_bounds = None
        self.lower_bounds = None
        self.task = None
        self.coef = coef
        self.bias = bias

    def fit(self, X, y, weight=None, task=None):
        self.task = task
        if task == 'reg':
            self.upper_bounds = y.max()
            self.lower_bounds = y.min()
        self.model.fit(X, y, weight)

    def predict(self, X):
        if self.task == 'reg':
            if X.ndim == 2:
                return np.clip(self.model.predict(X), self.lower_bounds, self.upper_bounds)
            else:
                return np.clip(self.model.predict(X.reshape(1, -1)), self.lower_bounds, self.upper_bounds)
        else:
            if X.ndim == 2:
                return self.model.predict(X)
            else:
                return self.model.predict(X.reshape(1, -1))


class elm_clip:
    def __init__(self, alpha=None, hidden_layer_nodes=None):
        self.alpha = alpha
        self.hidden_layer_nodes = hidden_layer_nodes
        self.upper_bounds = None
        self.lower_bounds = None
        self.task = None
        self.scaler = StandardScaler()

    def fit(self, Xn, yn, weight=None, task=None):
        X = np.copy(Xn)
        y = np.copy(yn)
        self.task = task
        if task == 'reg':
            self.upper_bounds = np.max(y)
            self.lower_bounds = np.min(y)
        input_dimension = X.shape[1]
        self.a = np.random.normal(0, 1, (input_dimension, self.hidden_layer_nodes))
        self.b = np.random.normal(0, 1)
        H = X.dot(self.a) + self.b
        H = self.scaler.fit_transform(H)
        H = expit(H)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(H, y, weight)

    def predict(self, Xn):
        X = np.copy(Xn)
        if X.ndim != 2:
            X = X.reshape(1, -1)
        H = X.dot(self.a) + self.b
        H = self.scaler.transform(H)
        H = expit(H)
        if self.task == 'reg':
            if X.ndim == 2:
                return np.clip(self.model.predict(H), self.lower_bounds, self.upper_bounds)
            else:
                return np.clip(self.model.predict(H.reshape(1, -1)), self.lower_bounds, self.upper_bounds)
        else:
            if X.ndim == 2:
                return self.model.predict(H)
            else:
                return self.model.predict(H.reshape(1, -1))


class BoostTree(BaseEstimator):
    def __init__(self, max_leafs=5, node_model='Ridge', min_sample_leaf_list=None, reg_alpha_list=None, max_depth=None,
                 elm_hidden_layer_nodes=None, random_state=0):
        if min_sample_leaf_list is None:
            min_sample_leaf_list = 1
        if reg_alpha_list is None:
            reg_alpha_list = 0.1
        if elm_hidden_layer_nodes is None:
            elm_hidden_layer_nodes = 100
        self.max_leafs = max_leafs
        self.node_model = node_model
        self.min_sample_leaf_list = min_sample_leaf_list
        self.reg_alpha_list = reg_alpha_list
        self.max_depth = max_depth
        self.random_state = random_state
        self.elm_hidden_layer_nodes = elm_hidden_layer_nodes

    def save_model(self, file):
        """
        Parameters
        ----------
        file: str
            Controls the filename.
        """
        check_is_fitted(self, ['tree_'])
        joblib.dump(self, filename=file)

    def _create_node(self, output, depth, container, loss_node, model_node=None, sample_index=None):
        node = {
            "sample_index": sample_index,
            "index": container["index_node_global"],
            "loss": loss_node,
            "model": model_node,
            "output": output,
            "n_samples": len(sample_index),
            "split_feature": int,
            "threshold": float,
            "children": {
                "left": None,
                "right": None
            },
            "depth": depth,
        }
        container["index_node_global"] += 1
        return node

    def _check_max_leaf(self):
        if self.max_leafs is None:
            return True
        else:
            return self.leaf_num_ < self.max_leafs

    def _check_max_depth(self, depth):
        if self.max_depth is None:
            return True
        else:
            return depth < self.max_depth

    def _split_loss(self, f_t, grad, hess):
        _, _, idx_left, idx_right = f_t
        loss_split = self._calObj(grad[idx_left], hess[idx_left], self.reg_alpha_) + self._calObj(grad[idx_right], hess[idx_right],
                                                                                                  self.reg_alpha_)
        return loss_split

    @staticmethod
    def _find_feature_threshold(X, min_samples_leaf):
        feature_threshold = []
        total_index = np.arange(len(X))
        for split_feature in range(X.shape[1]):
            value_min, value_max = min(X[:, split_feature]), max(X[:, split_feature])
            threshold = np.random.uniform(value_min, value_max)
            idx = (X[:, split_feature] <= threshold)
            idx_left, idx_right = total_index[idx], total_index[~idx]
            if (len(idx_left) >= min_samples_leaf) and (len(idx_right) >= min_samples_leaf):
                feature_threshold.append((split_feature, threshold, idx_left, idx_right))
        return feature_threshold

    def _split_traverse_node(self, node, container, node_temp, loss_temp, task):
        did_split = self._splitter(node, container, task)
        del node["output"], node["sample_index"]
        loss_temp[node["index"]] = 0
        # Return terminal node if split is not advised
        if not did_split:
            if not self._check_max_leaf():
                return node_temp, loss_temp, -1
            if max(loss_temp) == 0:
                return node_temp, loss_temp, -1
            split_index = loss_temp.index(max(loss_temp))
            return node_temp, loss_temp, split_index

        node_temp.append(node["children"]["left"])
        loss_temp.append(node["children"]["left"]["n_samples"] * node["children"]["left"]["loss"])
        node_temp.append(node["children"]["right"])
        loss_temp.append(node["children"]["right"]["n_samples"] * node["children"]["right"]["loss"])

        self.leaf_num_ += 1
        if max(loss_temp) == 0:
            return node_temp, loss_temp, -1
        split_index = loss_temp.index(max(loss_temp))
        return node_temp, loss_temp, split_index

    def _splitter(self, node, container, task):
        output, sample_index, n_samples = node["output"], node["sample_index"], node["n_samples"]
        did_split = False
        if not self._check_max_leaf():
            return did_split
        if not self._check_max_depth(node["depth"]):
            return did_split
        if n_samples > 1000:
            mask = random.sample(range(n_samples), 1000)
            batch_index = sample_index[mask]
            X_split, y_split = self.train_X[batch_index], self.train_y[batch_index]
            if output is not None:
                output_split = output[mask]
            else:
                output_split = self._predict_score(node['model'], X_split)
        else:
            if n_samples == len(self.train_X):
                X_split, y_split = self.train_X, self.train_y
            else:
                X_split, y_split = self.train_X[sample_index], self.train_y[sample_index]
            if output is not None:
                output_split = output
            else:
                output_split = self._predict_score(node['model'], X_split)
        grad = self._calGrad(output_split, y_split)
        hess = self._calHess(output_split, y_split)
        if len(X_split) < 2 * self.min_samples_leaf_:
            return did_split
        feature_threshold = self._find_feature_threshold(X_split, self.min_samples_leaf_)
        if node['index'] == 0 and len(feature_threshold) == 0:
            while True:
                feature_threshold = self._find_feature_threshold(X_split, self.min_samples_leaf_)
                if len(feature_threshold) > 0:
                    break
        if len(feature_threshold) == 0:
            return did_split
        did_split = True
        loss_list = [self._split_loss(f_t, grad, hess) for f_t in feature_threshold]
        best_loss_split = min(loss_list)
        best_loss_index = loss_list.index(best_loss_split)
        split_feature, threshold, _, _ = feature_threshold[best_loss_index]
        if n_samples == len(self.train_X):
            train_X_ = self.train_X[:, split_feature]
        else:
            train_X_ = self.train_X[:, split_feature][sample_index]
        idx_true = (train_X_ <= threshold)
        idx_left = sample_index[idx_true]
        idx_right = sample_index[~idx_true]
        if n_samples > 1000:
            loss_left, model_left, output_left_ = self._fit_model(idx_left, idx_true, output=output, premodel=node['model'])
            loss_right, model_right, output_right_ = self._fit_model(idx_right, ~idx_true, output=output, premodel=node['model'])
            if len(idx_left) > 1000:
                if task == 'reg':
                    output_left = output[idx_true] + self._predict_score(model_left, self.train_X[idx_left])
                else:
                    output_left = None
            else:
                output_left = output_left_
            if len(idx_right) > 1000:
                if task == 'reg':
                    output_right = output[~idx_true] + self._predict_score(model_right, self.train_X[idx_right])
                else:
                    output_right = None
            else:
                output_right = output_right_
        else:
            loss_left, model_left, output_left = self._fit_model(idx_left, idx_true, output=output_split, premodel=node['model'], task=task)
            loss_right, model_right, output_right = self._fit_model(idx_right, ~idx_true, output=output_split, premodel=node['model'], task=task)
        node["children"]["left"] = self._create_node(output_left, node["depth"] + 1, container, loss_left, model_left, idx_left)
        node["children"]["right"] = self._create_node(output_right, node["depth"] + 1, container, loss_right, model_right,
                                                      idx_right)
        node["split_feature"] = split_feature
        node["threshold"] = threshold
        if task == 'clf':
            del node['model']
        return did_split

    @staticmethod
    def _weight_and_response(y, prob):
        sample_weight = prob * (1. - prob)
        sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)
        with np.errstate(divide="ignore", over="ignore"):
            z = np.where(y, 1. / prob, -1. / (1. - prob))
        z = np.clip(z, a_min=-max_response, a_max=max_response)
        return sample_weight, z

    def _output2prob(self, output):
        if self.n_classes_ <= 2:
            prob = expit(output)
        else:
            prob = softmax(output, axis=1)
        return prob

    def _fit_model(self, idx, idx_true, output=None, premodel=None, task=None):
        if len(idx) > 1000:
            mask = random.sample(range(len(idx)), 1000)
            batch_left = idx[mask]
            if output is not None:
                loss_, model_, y_pred_ = self.__fit_model(self.train_X[batch_left], self.train_y[batch_left], output=output[idx_true][mask],
                                                          premodel=premodel, task=task)
            else:
                loss_, model_, y_pred_ = self.__fit_model(self.train_X[batch_left], self.train_y[batch_left], premodel=premodel, task=task)
        else:
            if output is not None:
                loss_, model_, y_pred_ = self.__fit_model(self.train_X[idx], self.train_y[idx], output=output[idx_true], premodel=premodel,
                                                          task=task)
            else:
                loss_, model_, y_pred_ = self.__fit_model(self.train_X[idx], self.train_y[idx], premodel=premodel, task=task)
        return loss_, model_, y_pred_

    def __fit_model(self, X, y, output=None, premodel=None, task=None):
        w, bias = None, None
        if self.node_model == 'Ridge':
            model = ridge_clip(self.reg_alpha_)
        elif self.node_model == 'ELM':
            model = elm_clip(self.reg_alpha_, self.elm_hidden_layer_nodes_)
        else:
            model = ridge_clip(self.reg_alpha_)
        if output is None:
            output = self._predict_score(premodel, X)
        if task == 'clf':
            prob = self._output2prob(output)
            if self.n_classes_ <= 2:
                weight, z = self._weight_and_response(y, prob)
                # X_train, z_train, weight_train = filter_quantile(X, z, weight, trim_quantile=0.05)
                X_train, z_train, weight_train = X, z, weight
                new_estimators = deepcopy(model)
                new_estimators.fit(X_train, z_train, weight_train)

            else:
                new_estimators = []
                for j in range(self.n_classes_):
                    weight, z = self._weight_and_response(y[:, j], prob[:, j])
                    X_train, z_train, weight_train = X, z, weight
                    model_copy = deepcopy(model)
                    model_copy.fit(X_train, z_train, weight_train)
                    new_estimators.append(model_copy)
                w = np.array([e.model.coef_ for e in new_estimators]).T
                bias = np.array([e.model.intercept_ for e in new_estimators]).T
            if premodel is not None:
                if self.n_classes_ <= 2:
                    new_estimators.model.coef_ += premodel.model.coef_
                    new_estimators.model.intercept_ += premodel.model.intercept_
                else:
                    w += premodel.coef
                    bias += premodel.bias
            if self.n_classes_ > 2:
                new_estimators = ridge_clip(coef=w, bias=bias)
            y_pred = self._predict_score(new_estimators, X)
            prob = self._output2prob(y_pred)
            loss = self._loss(y, prob)
        else:
            new_estimators = deepcopy(model)
            new_estimators.fit(X, y - output, task='reg')
            y_pred = self._predict_score(new_estimators, X) + output
            loss = self._loss(y, y_pred)
        assert loss >= 0.0
        return loss, new_estimators, y_pred


class BoostTreeRegressor(RegressorMixin, BoostTree):

    @staticmethod
    def _predict_score(node_model, X):
        return node_model.predict(X)

    @staticmethod
    def _loss(y, y_pred):
        return mean_squared_error(y, y_pred)

    @staticmethod
    def _calGrad(y_pred, y_data):
        return y_pred - y_data

    @staticmethod
    def _calHess(y_pred, y_data):
        return np.ones_like(y_data)

    @staticmethod
    def _calObj(gard, hess, alpha=None):
        return (-1.0 / 2) * sum(gard) ** 2 / (sum(hess) + alpha)

    def _predict_x(self, node, x, y_pred_x=None):
        no_children = node["children"]["left"] is None and node["children"]["right"] is None
        if y_pred_x is None:
            y_pred_x = 0
        else:
            y_pred_x += self._predict_score(node["model"], x)
        if no_children:
            return y_pred_x
        else:
            if x[node["split_feature"]] <= node["threshold"]:
                return self._predict_x(node["children"]["left"], x, y_pred_x)
            else:
                return self._predict_x(node["children"]["right"], x, y_pred_x)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        y = column_or_1d(y, warn=True)
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        self.leaf_num_ = 1
        if isinstance(self.min_sample_leaf_list, list):
            self.min_samples_leaf_ = np.random.choice(self.min_sample_leaf_list, 1)[0]
        else:
            self.min_samples_leaf_ = self.min_sample_leaf_list
        if isinstance(self.reg_alpha_list, list):
            self.reg_alpha_ = np.random.choice(self.reg_alpha_list, 1)[0]
        else:
            self.reg_alpha_ = self.reg_alpha_list
        if isinstance(self.elm_hidden_layer_nodes, list):
            self.elm_hidden_layer_nodes_ = np.random.choice(self.elm_hidden_layer_nodes, 1)[0]
        else:
            self.elm_hidden_layer_nodes_ = self.elm_hidden_layer_nodes
        self.train_X = X
        self.train_y = y
        n_samples = len(self.train_X)
        output = np.zeros(n_samples, dtype=np.float64)
        container = {"index_node_global": 0}
        self.tree_ = self._create_node(output, 0, container, 0, sample_index=np.arange(n_samples))
        node_temp = []
        node_temp.append(self.tree_)
        loss_temp = []
        loss_temp.append(self.tree_["n_samples"] * self.tree_["loss"])
        split_index = 0
        while split_index >= 0:
            node_temp, loss_temp, split_index = self._split_traverse_node(node_temp[split_index], container, node_temp, loss_temp, task='reg')
        while True:
            if max(loss_temp) == 0:
                break
            else:
                max_index = loss_temp.index(max(loss_temp))
                loss_temp[node_temp[max_index]["index"]] = 0
                del node_temp[max_index]["output"], node_temp[max_index]["sample_index"]
        del self.train_X, self.train_y
        return self

    def predict(self, X):
        check_is_fitted(self, ['tree_'])
        X = check_array(X)
        assert self.tree_ is not None
        y_pred = np.array([self._predict_x(self.tree_, x) for x in X])
        y_pred = y_pred.reshape(len(X))
        return y_pred


class BoostTreeClassifier(ClassifierMixin, BoostTree):

    def _predict_score(self, node_model, X):
        if self.n_classes_ <= 2:
            new_scores = node_model.predict(X)
        else:
            if X.ndim == 2:
                new_scores = np.dot(X, node_model.coef) + node_model.bias
            else:
                new_scores = np.dot(X.reshape(1, -1), node_model.coef) + node_model.bias
            new_scores -= new_scores.mean(keepdims=True, axis=1)
            new_scores *= (self.n_classes_ - 1) / self.n_classes_
        return new_scores

    def _loss(self, y, y_pred):
        if self.n_classes_ <= 2:
            loss = log_loss(y, y_pred, labels=[0, 1])
        else:
            loss = log_loss(y, y_pred, labels=np.eye(self.n_classes_))
        return loss

    def _calGrad(self, y_pred, y_data):
        prob = self._output2prob(y_pred)
        return prob - y_data

    def _calHess(self, y_pred, y_data):
        prob = self._output2prob(y_pred)
        return prob * (1 - prob)

    def _calObj(self, grad, hess, alpha=None):
        if self.n_classes_ > 2:
            loss_ = 0.0
            for j in range(self.n_classes_):
                loss_ += (-1.0 / 2) * sum(grad[:, j]) ** 2 / (sum(hess[:, j]) + alpha)
            return loss_
        else:
            return (-1.0 / 2) * sum(grad) ** 2 / (sum(hess) + alpha)

    def _predict_x(self, node, x, y_pred_x=None):
        no_children = node["children"]["left"] is None and node["children"]["right"] is None
        if no_children:
            return self._predict_score(node["model"], x)
        else:
            if x[node["split_feature"]] <= node["threshold"]:
                return self._predict_x(node["children"]["left"], x, y_pred_x)
            else:
                return self._predict_x(node["children"]["right"], x, y_pred_x)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.leaf_num_ = 1

        if isinstance(self.min_sample_leaf_list, list):
            self.min_samples_leaf_ = np.random.choice(self.min_sample_leaf_list, 1)[0]
        else:
            self.min_samples_leaf_ = self.min_sample_leaf_list
        if isinstance(self.reg_alpha_list, list):
            self.reg_alpha_ = np.random.choice(self.reg_alpha_list, 1)[0]
        else:
            self.reg_alpha_ = self.reg_alpha_list
        if isinstance(self.elm_hidden_layer_nodes, list):
            self.elm_hidden_layer_nodes_ = np.random.choice(self.elm_hidden_layer_nodes, 1)[0]
        else:
            self.elm_hidden_layer_nodes_ = self.elm_hidden_layer_nodes
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=0)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            _ = column_or_1d(y, warn=True)
        else:
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))
        if Y.shape[1] == 1:
            Y = Y.ravel()
        X, Y = check_X_y(X, Y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        self.n_classes_ = len(self.classes_)

        self.train_X = X
        self.train_y = Y
        n_samples = len(self.train_X)
        if self.n_classes_ <= 2:
            output = np.zeros(n_samples, dtype=np.float64)
        else:
            output = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        container = {"index_node_global": 0}
        self.tree_ = self._create_node(output, 0, container, 0, sample_index=np.arange(n_samples))
        node_temp = []
        node_temp.append(self.tree_)
        loss_temp = []
        loss_temp.append(self.tree_["n_samples"] * self.tree_["loss"])
        split_index = 0
        while split_index >= 0:
            node_temp, loss_temp, split_index = self._split_traverse_node(node_temp[split_index], container, node_temp, loss_temp, task='clf')
        while True:
            if max(loss_temp) == 0:
                break
            else:
                max_index = loss_temp.index(max(loss_temp))
                loss_temp[node_temp[max_index]["index"]] = 0
                del node_temp[max_index]["output"], node_temp[max_index]["sample_index"]
        del self.train_X, self.train_y
        return self

    @property
    def classes_(self):
        """
        Classes labels
        """
        return self._label_binarizer.classes_

    def predict(self, X):
        check_is_fitted(self, ['tree_'])
        X = check_array(X)
        assert self.tree_ is not None
        y_pred = np.array([self._predict_x(self.tree_, x) for x in X])
        scores = y_pred.reshape(len(X), -1)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        check_is_fitted(self, ['tree_'])
        X = check_array(X)
        assert self.tree_ is not None
        y_pred = np.array([self._predict_x(self.tree_, x) for x in X])
        scores = y_pred.reshape(len(X), -1)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if self.n_classes_ == 1:
            prob = expit(scores)
        elif self.n_classes_ == 2:
            prob = expit(scores)
            prob = np.c_[1 - prob, prob]
        else:
            prob = softmax(scores, axis=1)
        return prob


if __name__ == "__main__":
    # Check if meet Sklearn's manner
    # ------------------------------------
    check_estimator(BoostTreeRegressor())
    check_estimator(BoostTreeClassifier())
