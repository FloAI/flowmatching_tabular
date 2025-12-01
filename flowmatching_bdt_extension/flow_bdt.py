from sklearn.base import clone
from xgboost import XGBRegressor
from flowmatching_bdt.flow_matcher import ConditionalFlowMatcher
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm


def duplicate(arr, n_times):
    if len(arr.shape) == 1:
        arr = arr[:, None]
    return np.tile(arr, (n_times, 1))


class FlowMatchingBDT:
    def __init__(
        self,
        n_flow_steps=50,
        n_duplicates=100,
        base_regressor=None,
        # default XGBoost parameters if user wants XGB by default
        max_depth=7,
        n_estimators=100,
        eta=0.3,
        tree_method="approx",
        reg_lambda=0.0,
        reg_alpha=0.0,
        subsample=1.0,
    ):
        """
        base_regressor: any sklearn-style regressor, e.g.
            RandomForestRegressor()
            GradientBoostingRegressor()
            ExtraTreesRegressor()
            CatBoostRegressor()
            etc.

        If None → XGBRegressor is used.
        """
        if base_regressor is None:
            self.base_regressor = XGBRegressor(
                objective="reg:squarederror",
                max_depth=max_depth,
                n_estimators=n_estimators,
                eta=eta,
                tree_method=tree_method,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                subsample=subsample,
            )
        else:
            self.base_regressor = base_regressor

        self.n_flow_steps = n_flow_steps
        self.n_duplicates = n_duplicates

    def xt_and_vt(self, x1):
        n_samples, n_features = x1.shape
        x0 = np.random.normal(size=(n_samples, n_features))
        t_levels = np.linspace(1e-3, 1, self.n_flow_steps)

        X_train = np.zeros((self.n_flow_steps, n_samples, n_features))
        y_train = np.zeros((self.n_flow_steps, n_samples, n_features))

        FM = ConditionalFlowMatcher()

        for i in range(self.n_flow_steps):
            t = np.ones(n_samples) * t_levels[i]
            _, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)
            X_train[i], y_train[i] = xt, ut

        return X_train, y_train

    def train_single(self, xt, vt, conditions=None):
        # clone the regressor for this flow step
        model = clone(self.base_regressor)

        if conditions is not None:
            if conditions.ndim == 1:
                conditions = conditions[:, None]
            xt = np.concatenate([xt, conditions], axis=1)

        model.fit(xt, vt)
        return model

    def train(self, x_train, conditions=None):
        x1 = duplicate(x_train, self.n_duplicates)
        if conditions is not None:
            conditions = duplicate(conditions, self.n_duplicates)

        xt, vt = self.xt_and_vt(x1)

        def train_noise_level(i):
            return self.train_single(xt[i], vt[i], conditions)

        with tqdm_joblib(tqdm(total=self.n_flow_steps, desc="Training")):
            models = Parallel(n_jobs=-1)(
                delayed(train_noise_level)(i)
                for i in range(self.n_flow_steps)
            )
        return models

    def fit(self, x_train, conditions=None):
        self.n_features = x_train.shape[1]
        self.models = self.train(x_train, conditions)

    def model_t(self, t, xt, conditions=None):
        idx = int(round(t * (self.n_flow_steps - 1)))
        if conditions is not None:
            if conditions.ndim == 1:
                conditions = conditions[:, None]
            xt = np.concatenate([xt, conditions], axis=1)
        return self.models[idx].predict(xt)

    def euler_solve(self, x0, conditions=None, n_steps=100):
        h = 1 / (n_steps - 1)
        x = x0
        t = 0.0
        for _ in range(n_steps - 1):
            x = x + h * self.model_t(t, x, conditions)
            t += h
        return x

    def predict(self, num_samples, conditions=None):
        x0 = np.random.normal(size=(num_samples, self.n_features))
        return self.euler_solve(x0, conditions)
