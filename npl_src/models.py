# src/models.py
import numpy as np

class StackedResidualModel:
    def __init__(self, booster_lgb=None, model_ridge=None, model_cat=None, weights=None):
        self.booster_lgb = booster_lgb
        self.model_ridge = model_ridge
        self.model_cat = model_cat
        self.weights = weights or {}

    def predict(self, X, num_iteration=None):
        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        parts, w = [], (self.weights or {})
        if w.get("lgb", 0.0) > 0 and self.booster_lgb is not None:
            parts.append(w["lgb"] * self.booster_lgb.predict(X_arr, num_iteration=num_iteration))
        if w.get("ridge", 0.0) > 0 and self.model_ridge is not None:
            parts.append(w["ridge"] * self.model_ridge.predict(X_arr))
        if w.get("cat", 0.0) > 0 and self.model_cat is not None:
            parts.append(w["cat"] * self.model_cat.predict(X_arr))
        if not parts:
            return np.zeros(X_arr.shape[0], dtype=float)
        return np.sum(parts, axis=0)
