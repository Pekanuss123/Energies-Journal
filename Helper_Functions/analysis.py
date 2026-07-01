"""
analysis.py
-----------
Analysis routines for the Journal Paper experiments (Sections 5.1 – 5.4 + §5.6
persistence baseline).

Functions provided:

  Section 5.1 — Training-set size experiment
    training_size_sweep(df, target_col, model_fn_map, sizes, ...)
        → DataFrame  [size, model_name, MQL_sum]

  Section 5.2 — Model aggregation & generalisation (Individual / Grid / LOO)
    aggregation_scenarios(dfs, target_col, forecast_fn, ...)
        → DataFrame  [prosumer, scenario, MQL_sum]

  Section 5.3 — Auto-correlation analysis
    autocorrelation_matrix(dfs, target_col, lags, ...)
        → 2-D DataFrame  (lag × prosumer)  of ACF values
    plot_autocorrelation_heatmap(acf_matrix, title, ax)

  Section 5.4 — Feature-target correlation
    feature_target_correlation(dfs, target_col, feature_cols, ...)
        → dict  {method: DataFrame (feature × prosumer)}
    plot_correlation_heatmap(corr_dict, title, ...)

  §5.6 — Persistence baseline
    persistence_forecast(y_test, forecast_steps)
        → predictions, lower_bounds, upper_bounds, y_test
          (all shape (n_samples, forecast_steps))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ===========================================================================
# Section 5.1 — Training-Set Size Sweep
# ===========================================================================

def training_size_sweep(
    df: pd.DataFrame,
    target_col: str,
    model_fn_map: dict,
    sizes: list = None,
    test_fraction: float = 0.20,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> pd.DataFrame:
    """Evaluate each model in *model_fn_map* across training sizes.

    Parameters
    ----------
    df : single-prosumer DataFrame with feature columns + target_col
    target_col : name of the target column
    model_fn_map : dict {model_name: callable}
        Each callable must match the interface used by models_conference.py:
            fn(X_train, y_train, X_test, y_test, scaler_y, lower_q, upper_q)
            → dict with keys y_pred, y_lower, y_upper
    sizes : list of integer training-set sizes to test;
            defaults to [50,100,200,300,400,500,1000,1500,...,9000]
    test_fraction : fraction of the FULL dataset held out as test set.
    lower_q, upper_q : quantile bounds passed to each model.

    Returns
    -------
    DataFrame with columns [size, model, MQL_sum]
    """
    if sizes is None:
        sizes = [50, 100, 200, 300, 400, 500,
                 1000, 1500, 2000, 2500, 3000, 3500,
                 4000, 4500, 5000, 5500, 6000, 6500,
                 7000, 7500, 8000, 8500, 9000]

    # Build feature + target arrays once with MinMaxScaler
    feature_cols = [c for c in df.columns if c != target_col]
    X_all = df[feature_cols].values
    y_all = df[target_col].values

    n_total = len(X_all)
    n_test  = int(n_total * test_fraction)
    X_test_raw = X_all[n_total - n_test:]
    y_test_raw = y_all[n_total - n_test:]

    rows = []
    for size in sizes:
        if size > n_total - n_test:
            print(f'  [skip] size={size} exceeds available training rows')
            continue

        # Take the *last* `size` rows from the non-test portion
        start = (n_total - n_test) - size
        X_train_raw = X_all[start: n_total - n_test]
        y_train_raw = y_all[start: n_total - n_test]

        # Scale X
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(X_train_raw)
        X_test  = scaler_x.transform(X_test_raw)

        # Scale y
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
        y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

        for model_name, fn in model_fn_map.items():
            try:
                result = fn(X_train, y_train, X_test, y_test,
                            scaler_y, lower_q, upper_q)
                y_pred  = np.array(result['y_pred'])
                y_lower = np.array(result['y_lower'])
                y_upper = np.array(result['y_upper'])
                actuals = np.array(result['y_true'])

                mql = _mql_sum(y_lower, y_pred, y_upper, actuals,
                               lower_q, upper_q)
                rows.append({'size': size, 'model': model_name, 'MQL_sum': mql})
            except Exception as e:
                print(f'  [warn] {model_name} size={size}: {e}')
                rows.append({'size': size, 'model': model_name, 'MQL_sum': np.nan})

    return pd.DataFrame(rows)


def _mql_sum(lower, median, upper, actuals, lower_q=0.10, upper_q=0.90):
    """Pinball loss summed over lower / median / upper predictions."""
    def pinball(y, yhat, p):
        e = y - yhat
        return float(np.mean(np.maximum(p * e, (p - 1) * e)))
    return (pinball(actuals, lower,  lower_q) +
            pinball(actuals, median, 0.5) +
            pinball(actuals, upper,  upper_q)) / 3.0


def plot_training_size_sweep(sweep_df: pd.DataFrame,
                             title: str = 'MQL_Σ vs Training Set Size',
                             output_path: str = None) -> None:
    """Reproduce Figure 3 of the Journal Paper."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in sweep_df.groupby('model'):
        ax.plot(grp['size'], grp['MQL_sum'], marker='o', label=model)
    ax.set_xlabel('Training Set Size', fontsize=13)
    ax.set_ylabel('MQL_Σ', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ===========================================================================
# Section 5.2 — Model Aggregation & Generalisation
# ===========================================================================

def aggregation_scenarios(
    dfs: list,          # list of (name, DataFrame) for each Uppsala prosumer
    target_col: str,
    forecast_fn,        # e.g. lgbm_singlestep_forecast from models_conference
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> pd.DataFrame:
    """Run Individual / Grid / LOO scenarios and return MQL_sum per household.

    Parameters
    ----------
    dfs : list of (name, DataFrame) tuples, one per prosumer (Uppsala, 7 HH)
    target_col : 'Produced'
    forecast_fn : function matching models_conference signature
    lower_q, upper_q : quantile bounds

    Returns
    -------
    DataFrame  [prosumer, scenario, MQL_sum]
    """
    rows = []
    n = len(dfs)
    feature_cols = [c for c in dfs[0][1].columns if c != target_col]

    # Helper: scale and split one DataFrame
    def _prep(df):
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = df[feature_cols].values
        y = df[target_col].values
        split = int(len(df) * 0.80)
        X_tr = scaler_x.fit_transform(X[:split])
        X_te = scaler_x.transform(X[split:])
        y_tr = scaler_y.fit_transform(y[:split].reshape(-1, 1)).ravel()
        y_te = scaler_y.transform(y[split:].reshape(-1, 1)).ravel()
        return X_tr, y_tr, X_te, y_te, scaler_y

    # --- Individual: train on own data, test on own test set ---
    for name, df in dfs:
        X_tr, y_tr, X_te, y_te, scaler_y = _prep(df)
        res = forecast_fn(X_tr, y_tr, X_te, y_te, scaler_y, lower_q, upper_q)
        mql = _mql_sum(np.array(res['y_lower']), np.array(res['y_pred']),
                       np.array(res['y_upper']), np.array(res['y_true']),
                       lower_q, upper_q)
        rows.append({'prosumer': name, 'scenario': 'Individual', 'MQL_sum': mql})

    # --- Grid: train on all prosumers combined, test on each ---
    combined_df = pd.concat([df for _, df in dfs], ignore_index=True)
    scaler_x_g = MinMaxScaler()
    scaler_y_g = MinMaxScaler()
    X_all = combined_df[feature_cols].values
    y_all = combined_df[target_col].values
    split_g = int(len(combined_df) * 0.80)
    X_tr_g = scaler_x_g.fit_transform(X_all[:split_g])
    y_tr_g = scaler_y_g.fit_transform(y_all[:split_g].reshape(-1, 1)).ravel()

    for name, df in dfs:
        X_te = scaler_x_g.transform(df[feature_cols].values[int(len(df) * 0.80):])
        y_te = scaler_y_g.transform(
            df[target_col].values[int(len(df) * 0.80):].reshape(-1, 1)
        ).ravel()
        res = forecast_fn(X_tr_g, y_tr_g, X_te, y_te, scaler_y_g, lower_q, upper_q)
        mql = _mql_sum(np.array(res['y_lower']), np.array(res['y_pred']),
                       np.array(res['y_upper']), np.array(res['y_true']),
                       lower_q, upper_q)
        rows.append({'prosumer': name, 'scenario': 'Grid', 'MQL_sum': mql})

    # --- LOO: train on all-but-one prosumers, test on the held-out ---
    for i, (name, df_held_out) in enumerate(dfs):
        train_parts = [d for j, (_, d) in enumerate(dfs) if j != i]
        train_df = pd.concat(train_parts, ignore_index=True)
        split_t  = int(len(train_df) * 0.80)
        scaler_x_l = MinMaxScaler()
        scaler_y_l = MinMaxScaler()
        X_tr_l = scaler_x_l.fit_transform(
            train_df[feature_cols].values[:split_t]
        )
        y_tr_l = scaler_y_l.fit_transform(
            train_df[target_col].values[:split_t].reshape(-1, 1)
        ).ravel()
        # Test on the held-out prosumer (full dataset)
        X_te_l = scaler_x_l.transform(df_held_out[feature_cols].values)
        y_te_l = scaler_y_l.transform(
            df_held_out[target_col].values.reshape(-1, 1)
        ).ravel()
        res = forecast_fn(X_tr_l, y_tr_l, X_te_l, y_te_l,
                          scaler_y_l, lower_q, upper_q)
        mql = _mql_sum(np.array(res['y_lower']), np.array(res['y_pred']),
                       np.array(res['y_upper']), np.array(res['y_true']),
                       lower_q, upper_q)
        rows.append({'prosumer': name, 'scenario': 'LOO', 'MQL_sum': mql})

    return pd.DataFrame(rows)


def plot_aggregation_scenarios(agg_df: pd.DataFrame,
                               title: str = 'MQL_Σ by Aggregation Scenario',
                               output_path: str = None) -> None:
    """Reproduce Figure 4 of the Journal Paper (boxplots per scenario)."""
    import matplotlib.ticker as mticker

    scenarios = ['Individual', 'Grid', 'LOO']
    data = [agg_df.loc[agg_df['scenario'] == s, 'MQL_sum'].values
            for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=scenarios, patch_artist=True)
    colors = ['#4878d0', '#ee854a', '#6acc65']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('MQL_Σ', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis='y')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ===========================================================================
# Section 5.3 — Auto-Correlation Analysis
# ===========================================================================

def autocorrelation_matrix(
    dfs: list,
    target_col: str,
    lags: int = 24,
) -> pd.DataFrame:
    """Compute ACF at lags 1…lags for each prosumer.

    Parameters
    ----------
    dfs : list of (name, DataFrame) tuples
    target_col : column to compute ACF on (e.g. 'Produced')
    lags : number of lags (default 24)

    Returns
    -------
    DataFrame  shape (lags, n_prosumers)
    """
    result = {}
    for name, df in dfs:
        series = df[target_col].dropna().values
        n = len(series)
        mu = series.mean()
        var = np.var(series, ddof=0)
        acf_vals = []
        for lag in range(1, lags + 1):
            cov = np.mean((series[:n - lag] - mu) * (series[lag:] - mu))
            acf_vals.append(cov / (var + 1e-12))
        result[name] = acf_vals

    index = [f'lag_{k}' for k in range(1, lags + 1)]
    return pd.DataFrame(result, index=index)


def plot_autocorrelation_heatmap(acf_matrix: pd.DataFrame,
                                 title: str = 'Auto-Correlation',
                                 output_path: str = None,
                                 ax=None) -> None:
    """Reproduce Figure 5 of the Journal Paper (heatmap)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(acf_matrix.T.values, aspect='auto', cmap='coolwarm',
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(acf_matrix.index)))
    ax.set_xticklabels([l.replace('lag_', '') for l in acf_matrix.index],
                       fontsize=9)
    ax.set_yticks(range(len(acf_matrix.columns)))
    ax.set_yticklabels(acf_matrix.columns, fontsize=9)
    ax.set_xlabel('Lag (hours)', fontsize=12)
    ax.set_title(title, fontsize=13)

    if own_fig:
        plt.colorbar(im, ax=ax, label='ACF')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()
    return im


# ===========================================================================
# Section 5.4 — Feature-Target Correlation
# ===========================================================================

def feature_target_correlation(
    dfs: list,
    target_col: str,
    feature_cols: list,
    methods: list = ('pearson', 'spearman', 'kendall'),
) -> dict:
    """Compute Pearson, Spearman, and Kendall τ between each feature and the
    target, averaged over all prosumers in *dfs*.

    Parameters
    ----------
    dfs : list of (name, DataFrame) — all prosumers for one site/target
    target_col : target column name (e.g. 'Produced', 'Total Consumed')
    feature_cols : list of feature column names to correlate
    methods : subset of ('pearson', 'spearman', 'kendall')

    Returns
    -------
    dict {method: DataFrame(features × prosumers)}
    Also includes a 'mean' key with average correlation per feature.
    """
    corr_fn = {
        'pearson':  lambda x, y: pearsonr(x, y)[0],
        'spearman': lambda x, y: spearmanr(x, y)[0],
        'kendall':  lambda x, y: kendalltau(x, y)[0],
    }

    results = {m: {} for m in methods}

    for name, df in dfs:
        target = df[target_col].dropna()
        for method in methods:
            corr_vals = {}
            for feat in feature_cols:
                if feat not in df.columns:
                    corr_vals[feat] = np.nan
                    continue
                feat_series = df.loc[target.index, feat].dropna()
                common = feat_series.index.intersection(target.index)
                if len(common) < 5:
                    corr_vals[feat] = np.nan
                    continue
                try:
                    corr_vals[feat] = corr_fn[method](
                        feat_series.loc[common].values,
                        target.loc[common].values,
                    )
                except Exception:
                    corr_vals[feat] = np.nan
            results[method][name] = pd.Series(corr_vals, index=feature_cols)

    # Build one DataFrame per method + an average column
    output = {}
    for method in methods:
        df_out = pd.DataFrame(results[method])   # features × prosumers
        df_out['mean'] = df_out.mean(axis=1)
        output[method] = df_out

    return output


def plot_correlation_heatmap(
    corr_dict: dict,
    title: str = 'Feature-Target Correlation',
    col_key: str = 'mean',
    output_path: str = None,
) -> None:
    """Reproduce Figure 6 of the Journal Paper.

    Plots a heatmap for each method (Pearson / Spearman / Kendall τ) using
    the *col_key* column (default 'mean' across prosumers).

    Parameters
    ----------
    corr_dict : output of feature_target_correlation()
    col_key   : which column to visualise ('mean' or a specific prosumer name)
    """
    methods = list(corr_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        df = corr_dict[method]
        vals = df[[col_key]].values        # shape (n_features, 1)
        im = ax.imshow(vals, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks([0])
        ax.set_xticklabels([col_key], fontsize=10)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=9)
        ax.set_title(f'{method.capitalize()}  ({title})', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ===========================================================================
# Section 5.6 — Persistence Baseline
# ===========================================================================

def persistence_forecast(
    y_test: np.ndarray,
    y_last_obs: np.ndarray,
    forecast_steps: int = 48,
    confidence_fraction: float = 0.90,
) -> tuple:
    """Naive persistence baseline: ŷ_{t+h} = y_t for all h.

    Parameters
    ----------
    y_test     : shape (n_samples, forecast_steps) — true future values
    y_last_obs : shape (n_samples,) — last observed value before each window
                 (5th element of the tuples returned by the preprocessing
                 functions; not y_test[:, 0] which is a future label)

    Returns
    -------
    predictions, lower_bounds, upper_bounds, y_test
        all shape (n_samples, forecast_steps)
    """
    last_obs  = y_last_obs.reshape(-1, 1)
    preds     = np.tile(last_obs, (1, forecast_steps))
    residuals = y_test - preds
    alpha     = (1 - confidence_fraction) / 2
    lo_pct  = alpha * 100
    hi_pct  = (1 - alpha) * 100
    lowers = preds + np.percentile(residuals, lo_pct,  axis=0)
    uppers = preds + np.percentile(residuals, hi_pct,  axis=0)

    return preds, lowers, uppers, y_test
