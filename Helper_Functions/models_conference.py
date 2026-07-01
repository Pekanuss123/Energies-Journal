"""
Conference Paper Baseline Models
---------------------------------
Single-step forecasting models used in the Conference Paper:
  - Gaussian Process (sklearn, RBF + WhiteKernel + DotProduct)
  - LightGBM single-step (3 quantile models: lower / median / upper)
  - CatBoost quantile regression (separate models per quantile)
  - CatBoost MultiQuantile (one model predicting all deciles at once)
  - Quantile Linear Regression (sklearn QuantileRegressor)

Each function follows the same interface:
    result = model_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                            lower_quantile=0.10, upper_quantile=0.90)

where result is a dict with keys:
    y_true, y_pred, y_lower, y_upper   – all 1-D numpy arrays (original scale)
    model                               – the fitted model (or dict of models)

The helper `create_conference_splits()` prepares the train/test split with
MinMaxScaling, matching the Conference Paper preprocessing pipeline.

Scenarios mirror the Conference Paper:
  - Scenario 1: Individual model per prosumer   (pass LOO_prosumer_idx=None)
  - Scenario 2: Model per Smart Grid            (pass the merged grid DataFrame)
  - Scenario 3: Model per Grid with LOO         (pass LOO_prosumer_idx=1..7)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.linear_model import QuantileRegressor
from lightgbm import LGBMRegressor
import catboost


# ---------------------------------------------------------------------------
# Preprocessing helper (Conference Paper style: MinMaxScaler, no windowing)
# ---------------------------------------------------------------------------

def create_conference_splits(df, target_column, test_size=0.2):
    """
    Prepare train/test splits for the Conference Paper pipeline.

    The split is performed chronologically (no shuffle) before scalers are
    fit.  Scalers are fitted on training data only.

    Parameters
    ----------
    df : pd.DataFrame
        Combined feature DataFrame (Date as index).
    target_column : str
        Name of the target column (e.g. 'Produced' or 'Total Consumed').
    test_size : float
        Fraction of data reserved for testing (default 0.20).

    Returns
    -------
    X_train, Y_train, X_test, Y_test : numpy arrays  (scaled)
    scaler_y : MinMaxScaler fitted on training Y only
    scaler_x : MinMaxScaler fitted on training X only
    """
    df = df.fillna(0)

    n_train = int(len(df) * (1.0 - test_size))
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    feature_cols = [c for c in df.columns if c != target_column]

    X_train_raw = df_train[feature_cols].values
    X_test_raw  = df_test[feature_cols].values
    Y_train_raw = df_train[target_column].values.reshape(-1, 1)
    Y_test_raw  = df_test[target_column].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    X_train  = scaler_x.fit_transform(X_train_raw)
    X_test   = scaler_x.transform(X_test_raw)

    scaler_y = MinMaxScaler()
    Y_train  = scaler_y.fit_transform(Y_train_raw)
    Y_test   = scaler_y.transform(Y_test_raw)

    return X_train, Y_train, X_test, Y_test, scaler_y, scaler_x


# ---------------------------------------------------------------------------
# Internal inverse-transform helper
# ---------------------------------------------------------------------------

def _inverse(scaler, arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Gaussian Process (sklearn: RBF + WhiteKernel + DotProduct)
# ---------------------------------------------------------------------------

def gp_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                lower_quantile=0.10, upper_quantile=0.90,
                pretrained_model=None):
    """
    Gaussian Process regression with uncertainty intervals.

    Uses sklearn GaussianProcessRegressor with the kernel
    RBF() + WhiteKernel() + DotProduct() as used in the Conference Paper.

    Parameters
    ----------
    X_train, Y_train : arrays  – scaled training data
    X_test, Y_test   : arrays  – scaled test data
    scaler_y         : fitted MinMaxScaler for the target
    lower_quantile, upper_quantile : float – PI quantiles (default 0.10 / 0.90)
    pretrained_model : GaussianProcessRegressor or None
        Pass a previously trained model to skip re-fitting (Grid / LOO
        scenarios where prosumers share one model).

    Returns
    -------
    dict with keys: y_true, y_pred, y_lower, y_upper, model
    """
    kernel = RBF() + WhiteKernel() + DotProduct()

    if pretrained_model is None:
        gp = GaussianProcessRegressor(
            kernel=kernel, copy_X_train=False, n_restarts_optimizer=3
        )
        gp.fit(X_train, Y_train.ravel())
    else:
        gp = pretrained_model

    y_pred_scaled, sigma = gp.predict(X_test, return_std=True)
    y_upper_scaled = norm.ppf(upper_quantile, loc=y_pred_scaled, scale=sigma)
    y_lower_scaled = norm.ppf(lower_quantile, loc=y_pred_scaled, scale=sigma)

    return {
        "y_true":  _inverse(scaler_y, Y_test),
        "y_pred":  _inverse(scaler_y, y_pred_scaled),
        "y_upper": _inverse(scaler_y, y_upper_scaled),
        "y_lower": _inverse(scaler_y, y_lower_scaled),
        "model":   gp,
    }


# ---------------------------------------------------------------------------
# LightGBM single-step (3 separate quantile models)
# ---------------------------------------------------------------------------

def lgbm_singlestep_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                              lower_quantile=0.10, upper_quantile=0.90,
                              pretrained_models=None):
    """
    Single-step LightGBM forecasting with quantile prediction intervals.

    Trains (or reuses) three separate LGBMRegressors with quantile objective:
    one for the lower bound, one for the median, one for the upper bound.

    Parameters
    ----------
    pretrained_models : dict or None
        Dict with keys 'lower', 'median', 'upper' containing fitted models.

    Returns
    -------
    dict with keys: y_true, y_pred, y_lower, y_upper, model
        where model = {'lower': lgb_lower, 'median': lgb_median, 'upper': lgb_upper}
    """
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'quantile',
        'metric': 'quantile',
        'n_jobs': 1,
        'max_depth': 4,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'verbose': -1,
    }

    Y_flat = Y_train.ravel()

    if pretrained_models is None:
        lgb_lower = LGBMRegressor(alpha=lower_quantile, **lgb_params)
        lgb_lower.fit(X_train, Y_flat)

        lgb_median = LGBMRegressor(alpha=0.5, **lgb_params)
        lgb_median.fit(X_train, Y_flat)

        lgb_upper = LGBMRegressor(alpha=upper_quantile, **lgb_params)
        lgb_upper.fit(X_train, Y_flat)
    else:
        lgb_lower = pretrained_models['lower']
        lgb_median = pretrained_models['median']
        lgb_upper = pretrained_models['upper']

    return {
        "y_true":  _inverse(scaler_y, Y_test),
        "y_pred":  _inverse(scaler_y, lgb_median.predict(X_test)),
        "y_lower": _inverse(scaler_y, lgb_lower.predict(X_test)),
        "y_upper": _inverse(scaler_y, lgb_upper.predict(X_test)),
        "model":   {'lower': lgb_lower, 'median': lgb_median, 'upper': lgb_upper},
    }


# ---------------------------------------------------------------------------
# CatBoost single-quantile (3 separate Quantile models)
# ---------------------------------------------------------------------------

def catboost_quantile_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                                lower_quantile=0.10, upper_quantile=0.90,
                                pretrained_models=None,
                                verbose=0):
    """
    CatBoost quantile regression with 3 separate models (lower / median / upper).

    Parameters
    ----------
    pretrained_models : dict or None
        Dict with keys 'lower', 'median', 'upper' containing fitted CatBoost models.
    verbose : int
        CatBoost verbosity (0 = silent).

    Returns
    -------
    dict with keys: y_true, y_pred, y_lower, y_upper, model
    """
    Y_flat = Y_train.ravel()

    if pretrained_models is None:
        cb_lower = catboost.CatBoostRegressor(
            loss_function=f'Quantile:alpha={lower_quantile}', verbose=verbose
        )
        cb_lower.fit(X_train, Y_flat)

        cb_median = catboost.CatBoostRegressor(
            loss_function='Quantile:alpha=0.5', verbose=verbose
        )
        cb_median.fit(X_train, Y_flat)

        cb_upper = catboost.CatBoostRegressor(
            loss_function=f'Quantile:alpha={upper_quantile}', verbose=verbose
        )
        cb_upper.fit(X_train, Y_flat)
    else:
        cb_lower = pretrained_models['lower']
        cb_median = pretrained_models['median']
        cb_upper = pretrained_models['upper']

    return {
        "y_true":  _inverse(scaler_y, Y_test),
        "y_pred":  _inverse(scaler_y, cb_median.predict(X_test)),
        "y_lower": _inverse(scaler_y, cb_lower.predict(X_test)),
        "y_upper": _inverse(scaler_y, cb_upper.predict(X_test)),
        "model":   {'lower': cb_lower, 'median': cb_median, 'upper': cb_upper},
    }


# ---------------------------------------------------------------------------
# CatBoost MultiQuantile (single model predicting all 9 deciles at once)
# ---------------------------------------------------------------------------

def catboost_multiquantile_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                                     lower_quantile=0.10, upper_quantile=0.90,
                                     pretrained_model=None,
                                     verbose=0):
    """
    CatBoost MultiQuantile regression: one model predicting all deciles 0.1–0.9.

    The lower bound is taken from decile index 0 (alpha=0.1),
    the median from index 4 (alpha=0.5),
    the upper bound from index 8 (alpha=0.9).

    Parameters
    ----------
    pretrained_model : CatBoostRegressor or None

    Returns
    -------
    dict with keys: y_true, y_pred, y_lower, y_upper, model
    """
    quantiles = [q / 10 for q in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
    quantile_str = str(quantiles).replace('[', '').replace(']', '')

    Y_flat = Y_train.ravel()

    if pretrained_model is None:
        cb = catboost.CatBoostRegressor(
            loss_function=f'MultiQuantile:alpha={quantile_str}', verbose=verbose
        )
        cb.fit(X_train, Y_flat)
    else:
        cb = pretrained_model

    preds = cb.predict(X_test)  # shape (n_samples, 9)
    y_lower_scaled = preds[:, 0]   # alpha=0.1
    y_median_scaled = preds[:, 4]  # alpha=0.5
    y_upper_scaled = preds[:, 8]   # alpha=0.9

    return {
        "y_true":  _inverse(scaler_y, Y_test),
        "y_pred":  _inverse(scaler_y, y_median_scaled),
        "y_lower": _inverse(scaler_y, y_lower_scaled),
        "y_upper": _inverse(scaler_y, y_upper_scaled),
        "model":   cb,
    }


# ---------------------------------------------------------------------------
# Quantile Linear Regression (sklearn QuantileRegressor)
# ---------------------------------------------------------------------------

def linear_quantile_forecast(X_train, Y_train, X_test, Y_test, scaler_y,
                              lower_quantile=0.10, upper_quantile=0.90,
                              pretrained_models=None):
    """
    Quantile linear regression baseline using sklearn QuantileRegressor.

    Parameters
    ----------
    pretrained_models : dict or None
        Dict with keys 'lower', 'median', 'upper'.

    Returns
    -------
    dict with keys: y_true, y_pred, y_lower, y_upper, model
    """
    Y_flat = Y_train.ravel()

    if pretrained_models is None:
        qr_lower = QuantileRegressor(
            quantile=lower_quantile, alpha=0.5, solver='highs'
        )
        qr_lower.fit(X_train, Y_flat)

        qr_median = QuantileRegressor(quantile=0.5, alpha=0.5, solver='highs')
        qr_median.fit(X_train, Y_flat)

        qr_upper = QuantileRegressor(
            quantile=upper_quantile, alpha=0.5, solver='highs'
        )
        qr_upper.fit(X_train, Y_flat)
    else:
        qr_lower = pretrained_models['lower']
        qr_median = pretrained_models['median']
        qr_upper = pretrained_models['upper']

    return {
        "y_true":  _inverse(scaler_y, Y_test),
        "y_pred":  _inverse(scaler_y, qr_median.predict(X_test)),
        "y_lower": _inverse(scaler_y, qr_lower.predict(X_test)),
        "y_upper": _inverse(scaler_y, qr_upper.predict(X_test)),
        "model":   {'lower': qr_lower, 'median': qr_median, 'upper': qr_upper},
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: run all Conference Paper models on one dataset split
# ---------------------------------------------------------------------------

def run_all_conference_models(X_train, Y_train, X_test, Y_test, scaler_y,
                               lower_quantile=0.10, upper_quantile=0.90,
                               run_gp=True):
    """
    Train and evaluate all Conference Paper models on a single data split.

    Parameters
    ----------
    run_gp : bool
        Set False to skip the GP (it can be slow on large datasets).

    Returns
    -------
    dict keyed by model name, each value is the result dict from the
    individual model function (y_true, y_pred, y_lower, y_upper, model).
    """
    results = {}

    if run_gp:
        results['GP'] = gp_forecast(
            X_train, Y_train, X_test, Y_test, scaler_y,
            lower_quantile, upper_quantile
        )

    results['LightGBM'] = lgbm_singlestep_forecast(
        X_train, Y_train, X_test, Y_test, scaler_y,
        lower_quantile, upper_quantile
    )
    results['CatBoost'] = catboost_quantile_forecast(
        X_train, Y_train, X_test, Y_test, scaler_y,
        lower_quantile, upper_quantile
    )
    results['CatBoost_Multi'] = catboost_multiquantile_forecast(
        X_train, Y_train, X_test, Y_test, scaler_y,
        lower_quantile, upper_quantile
    )
    results['LinearQR'] = linear_quantile_forecast(
        X_train, Y_train, X_test, Y_test, scaler_y,
        lower_quantile, upper_quantile
    )

    return results
