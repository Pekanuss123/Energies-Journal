"""
evaluation.py
-------------
Saving forecast results, computing evaluation metrics, generating
error plots (MAE + MPIR over forecast horizon), and summary tables.

Metrics implemented:
  MAE   — Mean Absolute Error
  R²    — Coefficient of Determination
  MAPE  — Mean Absolute Percentage Error (zero-safe)
  SMQL  — Scaled Mean Quantile Loss
  PICP  — Prediction Interval Coverage Probability
  CFE   — Coverage Frequency Error  |q - PICP|
  MPIR  — Mean Prediction Interval Range
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_forecast_results(
    predictions: np.ndarray, lower_bounds: np.ndarray,
    upper_bounds: np.ndarray, actual_values: np.ndarray,
    output_file: str,
) -> None:
    """Persist forecast arrays to a wide-format CSV."""
    _, forecast_steps = predictions.shape
    data = {f'prediction_{i+1}':   predictions[:, i]   for i in range(forecast_steps)}
    data.update({f'lower_bound_{i+1}':  lower_bounds[:, i]  for i in range(forecast_steps)})
    data.update({f'upper_bound_{i+1}':  upper_bounds[:, i]  for i in range(forecast_steps)})
    data.update({f'actual_value_{i+1}': actual_values[:, i] for i in range(forecast_steps)})
    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f'Results saved → {output_file}')


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calculate_mape(predictions: np.ndarray, actuals: np.ndarray,
                   exclude_zeros: bool = True) -> float:
    """Standard MAPE (%), optionally excluding zero actual values."""
    if exclude_zeros:
        mask = actuals != 0
        predictions, actuals = predictions[mask], actuals[mask]
    eps = 1e-10
    actuals_safe = np.where(actuals == 0, eps, actuals)
    return float(np.mean(np.abs((actuals_safe - predictions) / actuals_safe)) * 100)


def calculate_smape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Symmetric MAPE (%).  Well-defined for negative and near-zero values.

    Formula (as used in the Journal Paper, Eq. 2):
        sMAPE = (1/N) * sum( |y - yhat| / ((|y| + |yhat|) / 2) ) * 100

    Samples where both actual and predicted are exactly zero are excluded
    to avoid 0/0.
    """
    denom = (np.abs(actuals) + np.abs(predictions)) / 2.0
    mask = denom > 0
    return float(np.mean(np.abs(actuals[mask] - predictions[mask]) / denom[mask]) * 100)


def calculate_mql_sum(lower: np.ndarray, median: np.ndarray,
                     upper: np.ndarray, actuals: np.ndarray,
                     lower_q: float = 0.1, upper_q: float = 0.9) -> float:
    """Mean Quantile Loss summed over lower (q=0.10), median (q=0.50),
    and upper (q=0.90) predictions — MQL_Sigma as defined in the paper.

    Pinball loss:  L(y, yhat; p) = max(p*(y-yhat), (p-1)*(y-yhat))
    """
    def _pinball(y, yhat, p):
        e = y - yhat
        return float(np.mean(np.maximum(p * e, (p - 1) * e)))

    return (_pinball(actuals, lower,  lower_q) +
            _pinball(actuals, median, 0.5) +
            _pinball(actuals, upper,  upper_q)) / 3.0


def calculate_smql(predictions: np.ndarray, actuals: np.ndarray,
                   quantiles=(0.1, 0.5, 0.9)) -> float:
    """Scaled Mean Quantile Loss averaged over quantiles."""
    losses = []
    for q in quantiles:
        e = actuals - predictions
        losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    return float(np.mean(losses))


def calculate_picp(lower: np.ndarray, upper: np.ndarray,
                   actuals: np.ndarray):
    """Prediction Interval Coverage Probability + boolean mask."""
    within = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(within)), within


def calculate_mpir(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Range."""
    return float(np.mean(upper - lower))


# ---------------------------------------------------------------------------
# Per-file metrics table
# ---------------------------------------------------------------------------

def compute_metrics_for_file(
    file_path: str,
    forecast_steps: int = 48,
    coverage_target: float = 0.9,
) -> pd.DataFrame:
    """Compute per-horizon metrics for a single result CSV."""
    df = pd.read_csv(file_path)
    rows = []
    for i in range(1, forecast_steps + 1):
        preds   = df[f'prediction_{i}'].values
        actuals = df[f'actual_value_{i}'].values
        lowers  = df[f'lower_bound_{i}'].values
        uppers  = df[f'upper_bound_{i}'].values

        mae   = mean_absolute_error(actuals, preds)
        r2    = r2_score(actuals, preds)
        mape  = calculate_mape(preds, actuals)
        smape = calculate_smape(preds, actuals)
        smql  = calculate_smql(preds, actuals)
        mpir  = calculate_mpir(lowers, uppers)
        picp, _ = calculate_picp(lowers, uppers, actuals)
        cfe   = abs(coverage_target - picp)

        rows.append({
            'Time Step': i,
            'MAE': mae, 'R2': r2, 'MAPE': mape, 'sMAPE': smape,
            'SMQL': smql, 'PICP': picp, 'CFE': cfe, 'MPIR': mpir,
        })
    return pd.DataFrame(rows)


def compute_all_metrics(results_dir: str, output_dir: str,
                        forecast_steps: int = 48,
                        coverage_target: float = 0.9) -> None:
    """Compute metrics for every CSV in results_dir; save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(results_dir, fname)
        metrics_df = compute_metrics_for_file(fpath, forecast_steps, coverage_target)
        out = os.path.join(output_dir, f'metrics_{fname}')
        metrics_df.to_csv(out, index=False)
        print(f'Metrics saved → {out}')


# ---------------------------------------------------------------------------
# Summary table (average across horizon)
# ---------------------------------------------------------------------------

def build_summary_table(metrics_dir: str) -> pd.DataFrame:
    """Aggregate per-horizon metrics into a single summary row per file."""
    rows = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(metrics_dir, fname))
        row = {'File': fname.replace('metrics_', '').replace('.csv', '')}
        for col in ('MAE', 'R2', 'MAPE', 'sMAPE', 'SMQL', 'PICP', 'CFE', 'MPIR'):
            if col in df.columns:
                row[f'Avg_{col}'] = round(df[col].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error plots (MAE + MPIR over forecast horizon)
# ---------------------------------------------------------------------------

PLOT_CONFIG = {
    # (row_index_in_all_metrics_csv, output_filename)
    'U_C_Direct':    (1,  'U_C_Direct.png'),
    'U_C_Hybrid':    (2,  'U_C_Hybrid.png'),
    'U_C_Recursive': (3,  'U_C_Recursive.png'),
    'U_P_Direct':    (22, 'U_P_Direct.png'),
    'U_P_Hybrid':    (23, 'U_P_Hybrid.png'),
    'U_P_Recursive': (24, 'U_P_Recursive.png'),
    'H_C_Direct':    (52, 'H_C_Direct.png'),
    'H_C_Hybrid':    (53, 'H_C_Hybrid.png'),
    'H_C_Recursive': (54, 'H_C_Recursive.png'),
    'H_P_Direct':    (67, 'H_P_Direct.png'),
    'H_P_Hybrid':    (68, 'H_P_Hybrid.png'),
    'H_P_Recursive': (69, 'H_P_Recursive.png'),
}


def plot_mae_mpir(metrics_df: pd.DataFrame, title: str,
                  output_path: str, mae_scale: float = 1.0) -> None:
    """Dual-axis line plot of MAE and MPIR over forecast horizon steps.

    mae_scale divides the MAE values before plotting (default 1.0 = no
    scaling).  If you need to match a specific y-axis range, pass an explicit
    value — but always update the axis label accordingly so the figure is
    self-describing.
    """
    steps = list(range(1, 49))
    mae_vals  = [metrics_df.loc[metrics_df['Time Step'] == s, 'MAE'].values[0] / mae_scale for s in steps]
    mpir_vals = [metrics_df.loc[metrics_df['Time Step'] == s, 'MPIR'].values[0] for s in steps]

    mae_label = 'MAE' if mae_scale == 1.0 else f'MAE / {mae_scale}'

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, mae_vals, marker='o', color='tab:blue',
             markersize=6, linewidth=2, label='MAE')
    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel(mae_label, color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(steps, mpir_vals, marker='o', color='tab:orange',
             markersize=6, linewidth=2, label='MPIR')
    ax2.set_ylabel('MPIR', color='tab:orange', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=12)

    plt.title(title, fontsize=13)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Plot saved → {output_path}')


def generate_all_error_plots(metrics_dir: str, images_dir: str) -> None:
    """Generate MAE/MPIR plots for each forecast configuration."""
    os.makedirs(images_dir, exist_ok=True)
    for label, (_, fname) in PLOT_CONFIG.items():
        # Infer the metrics file from label
        parts = label.split('_')          # e.g. ['H', 'C', 'Direct']
        city  = 'Halmstad' if parts[0] == 'H' else 'Uppsala'
        mode  = 'Consumption' if parts[1] == 'C' else 'Production'
        strat = parts[2].lower()           # direct / hybrid / recursive

        mfile = os.path.join(
            metrics_dir,
            f'metrics_{strat}_forecast_{city.lower()}_{mode.lower()}_0.csv'
        )
        if not os.path.exists(mfile):
            print(f'  [skip] {mfile} not found')
            continue

        df = pd.read_csv(mfile)
        plot_mae_mpir(df, title=label,
                      output_path=os.path.join(images_dir, fname))


def plot_single_forecast(predictions: np.ndarray, lower_bounds: np.ndarray,
                         upper_bounds: np.ndarray, actuals: np.ndarray,
                         row_idx: int = 0, output_path: str = None) -> None:
    """Plot one sample row: actual vs. prediction with confidence interval."""
    steps = range(1, len(predictions) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, actuals,     color='steelblue',   lw=2, label='Actual')
    plt.plot(steps, predictions, color='darkorange',  lw=2, label='Prediction')
    plt.fill_between(steps, lower_bounds, upper_bounds,
                     color='gray', alpha=0.25, label='90% CI')
    plt.xlabel('Forecast Step')
    plt.ylabel('Energy (kWh)')
    plt.title(f'Forecast vs Actual — sample {row_idx}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()
