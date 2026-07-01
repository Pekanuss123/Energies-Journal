"""
models.py
---------
Three multi-step forecasting strategies using LightGBM with quantile
regression for prediction intervals (90% CI by default).

  - direct_multistep_forecast   : separate model per horizon step
  - recursive_multistep_forecast: single model, feeds own predictions back
  - hybrid_multistep_forecast   : direct strategy with recursive feature augmentation

All functions share the same signature:
    (X_train, y_train, X_test, y_test,
     input_steps, forecast_steps, confidence_interval)
    -> predictions, lower_bounds, upper_bounds, y_test
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ---------------------------------------------------------------------------
# Shared hyperparameter grid
# ---------------------------------------------------------------------------
DEFAULT_PARAM_GRID = {
    'num_leaves':    [31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators':  [100, 200],
}


def _tune(X_train: np.ndarray, y_train: np.ndarray,
          param_grid: dict = DEFAULT_PARAM_GRID) -> dict:
    """Grid-search on a LightGBM regressor using TimeSeriesSplit; returns best params."""
    base = lgb.LGBMRegressor(device='cpu', verbose=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(base, param_grid, cv=tscv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_params_


def _quantile_bounds(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, params: dict,
                     alpha: float):
    """Return lower and upper quantile predictions."""
    lo_model = lgb.LGBMRegressor(
        objective='quantile', alpha=alpha, **params,
        device='cpu', verbose=-1)
    hi_model = lgb.LGBMRegressor(
        objective='quantile', alpha=1 - alpha, **params,
        device='cpu', verbose=-1)
    lo_model.fit(X_tr, y_tr)
    hi_model.fit(X_tr, y_tr)
    return lo_model.predict(X_te), hi_model.predict(X_te)


# ---------------------------------------------------------------------------
# 1. Direct multi-step forecast
# ---------------------------------------------------------------------------

def direct_multistep_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    input_steps: int = 24, forecast_steps: int = 48,
    confidence_interval: float = 0.9,
):
    """Train one independent model per forecast horizon step."""
    alpha = (1 - confidence_interval) / 2
    n_tr, n_t, n_f = X_train.shape
    n_te = X_test.shape[0]

    Xtr = X_train.reshape(n_tr, -1)
    Xte = X_test.reshape(n_te, -1)

    preds  = np.zeros((n_te, forecast_steps))
    lowers = np.zeros((n_te, forecast_steps))
    uppers = np.zeros((n_te, forecast_steps))

    for step in range(forecast_steps):
        print(f'  [Direct] step {step + 1}/{forecast_steps}')
        y_step = y_train[:, step]
        params = _tune(Xtr, y_step)

        model = lgb.LGBMRegressor(**params, device='cpu', verbose=-1)
        model.fit(Xtr, y_step)
        preds[:, step] = model.predict(Xte)

        lo, hi = _quantile_bounds(Xtr, y_step, Xte, params, alpha)
        lowers[:, step] = lo
        uppers[:, step] = hi

    return preds, lowers, uppers, y_test


# ---------------------------------------------------------------------------
# 2. Recursive multi-step forecast
# ---------------------------------------------------------------------------

def recursive_multistep_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    input_steps: int = 24, forecast_steps: int = 48,
    confidence_interval: float = 0.9,
):
    """Train one model on step-1, then feed predictions back recursively."""
    alpha = (1 - confidence_interval) / 2
    n_tr, n_t, n_f = X_train.shape
    n_te = X_test.shape[0]

    Xtr = X_train.reshape(n_tr, -1)
    Xte = X_test.reshape(n_te, -1)

    y_step = y_train[:, 0]
    params = _tune(Xtr, y_step)

    model = lgb.LGBMRegressor(**params, device='cpu', verbose=-1)
    model.fit(Xtr, y_step)

    lo_model = lgb.LGBMRegressor(
        objective='quantile', alpha=alpha, **params, device='cpu', verbose=-1)
    hi_model = lgb.LGBMRegressor(
        objective='quantile', alpha=1 - alpha, **params, device='cpu', verbose=-1)
    lo_model.fit(Xtr, y_step)
    hi_model.fit(Xtr, y_step)

    preds  = np.zeros((n_te, forecast_steps))
    lowers = np.zeros((n_te, forecast_steps))
    uppers = np.zeros((n_te, forecast_steps))

    Xte_rec = Xte.copy()
    for step in range(forecast_steps):
        print(f'  [Recursive] step {step + 1}/{forecast_steps}')
        preds[:, step]  = model.predict(Xte_rec)
        lowers[:, step] = lo_model.predict(Xte_rec)
        uppers[:, step] = hi_model.predict(Xte_rec)

        if step < forecast_steps - 1:
            Xte_rec = np.hstack([Xte_rec, preds[:, step].reshape(-1, 1)])
            Xte_rec = Xte_rec[:, -(input_steps * n_f):]

    return preds, lowers, uppers, y_test


# ---------------------------------------------------------------------------
# 3. Hybrid (Direct–Recursive) multi-step forecast
# ---------------------------------------------------------------------------

def hybrid_multistep_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    input_steps: int = 24, forecast_steps: int = 48,
    confidence_interval: float = 0.9,
):
    """Hybrid (DIRMO) multi-step forecast.

    Each step-h model is trained on the original lookback window augmented
    with predictions from already-trained lower-step models applied to
    X_train.  At test time the same augmentation uses predictions from the
    same lower-step models applied to X_test.
    """
    alpha = (1 - confidence_interval) / 2
    n_tr, n_t, n_f = X_train.shape
    n_te = X_test.shape[0]

    Xtr_aug = X_train.reshape(n_tr, -1)
    Xte_aug = X_test.reshape(n_te, -1)

    preds  = np.zeros((n_te, forecast_steps))
    lowers = np.zeros((n_te, forecast_steps))
    uppers = np.zeros((n_te, forecast_steps))

    for step in range(forecast_steps):
        print(f'  [Hybrid] step {step + 1}/{forecast_steps}')
        y_step = y_train[:, step]
        params = _tune(Xtr_aug, y_step)

        model = lgb.LGBMRegressor(**params, device='cpu', verbose=-1)
        model.fit(Xtr_aug, y_step)
        preds[:, step] = model.predict(Xte_aug)

        lo, hi = _quantile_bounds(Xtr_aug, y_step, Xte_aug, params, alpha)
        lowers[:, step] = lo
        uppers[:, step] = hi

        if step < forecast_steps - 1:
            train_preds_step = model.predict(Xtr_aug).reshape(-1, 1)
            test_preds_step  = preds[:, step].reshape(-1, 1)

            Xtr_aug = np.hstack([Xtr_aug, train_preds_step])
            Xte_aug = np.hstack([Xte_aug, test_preds_step])

            max_cols = input_steps * n_f
            if Xtr_aug.shape[1] > max_cols:
                Xtr_aug = Xtr_aug[:, -max_cols:]
                Xte_aug = Xte_aug[:, -max_cols:]

    return preds, lowers, uppers, y_test


# ---------------------------------------------------------------------------
# Convenience: run all three strategies for a dataset dict
# ---------------------------------------------------------------------------

def run_all_strategies(multistep_data: dict, results_dir: str = 'Results',
                       input_steps: int = 24, forecast_steps: int = 48,
                       confidence_interval: float = 0.9):
    """Run Direct, Recursive, and Hybrid on every prosumer and save results.

    Results are saved as CSVs to results_dir/ with names like:
        direct_forecast_halmstad_consumption_0.csv
    """
    import os
    from evaluation import save_forecast_results
    os.makedirs(results_dir, exist_ok=True)

    strategies = {
        'direct_forecast':    direct_multistep_forecast,
        'recursive_forecast': recursive_multistep_forecast,
        'hybrid_forecast':    hybrid_multistep_forecast,
    }

    for strategy_name, fn in strategies.items():
        print(f'\n=== {strategy_name.upper()} ===')
        for category, data_list in multistep_data.items():
            for idx, (Xtr, ytr, Xte, yte, _) in enumerate(data_list):
                print(f'  Prosumer: {category} [{idx}]')
                preds, lowers, uppers, y_actual = fn(
                    Xtr, ytr, Xte, yte,
                    input_steps=input_steps,
                    forecast_steps=forecast_steps,
                    confidence_interval=confidence_interval,
                )
                out = os.path.join(results_dir, f'{strategy_name}_{category}_{idx}.csv')
                save_forecast_results(preds, lowers, uppers, y_actual, out)
