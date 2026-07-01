"""
preprocessing.py
----------------
Data loading, merging, night-time filtering, feature engineering,
scaling, and multi-step window creation for prosumer energy forecasting.

Expected Dataset/ layout:
  Dataset/
    clients/          Raw per-prosumer CSVs (Uppsala_*.csv, Halmstad_*.csv)
    weather/          Upsalla_Weather.csv, Halmstad_Weather.csv,
                      Upsalla_rad.csv, Halmstad_rad.csv
    combined_datasets/  Merged prosumer+weather CSVs (written by prepare_datasets)
"""

import os
import glob
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Night-time filter (month-specific daylight hours for Sweden)
# ---------------------------------------------------------------------------
DAYLIGHT_HOURS = {
    1:  ( 8, 16),
    2:  ( 7, 17),
    3:  ( 6, 18),
    4:  ( 5, 20),
    5:  ( 4, 21),
    6:  ( 3, 22),
    7:  ( 3, 22),
    8:  ( 5, 21),
    9:  ( 6, 19),
    10: ( 7, 18),
    11: ( 7, 16),
    12: ( 8, 15),
}


def apply_nighttime_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove night-time rows (zero-production hours) from a DataFrame.

    The 'Date' column must be parseable as datetime.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    mask = pd.Series(False, index=df.index)
    for month, (h_start, h_end) in DAYLIGHT_HOURS.items():
        mask |= (
            (df['Date'].dt.month == month)
            & (df['Date'].dt.hour >= h_start)
            & (df['Date'].dt.hour <= h_end)
        )
    return df[mask]


# ---------------------------------------------------------------------------
# Load raw prosumer + weather data and build combined datasets
# ---------------------------------------------------------------------------

def load_prosumer_data(dataset_dir: str):
    """Load raw per-prosumer CSVs from Dataset/clients/.

    Returns:
        Uppsala  (list[pd.DataFrame])
        Halmstad (list[pd.DataFrame])
    """
    client_dir = os.path.join(dataset_dir, 'clients')

    ups_ids   = ['1935', '1957', '2049', '2679', '517', '605', '742']
    hal_ids   = ['1329', '2597', '2705', '2729', '918']

    Uppsala  = [pd.read_csv(os.path.join(client_dir, f'Uppsala_{i}.csv'))  for i in ups_ids]
    Halmstad = [pd.read_csv(os.path.join(client_dir, f'Halmstad_{i}.csv')) for i in hal_ids]

    for i, df in enumerate(Uppsala):
        Uppsala[i] = df.rename(columns={
            'bought': f'Bought_{i}', 'produced': f'Produced_{i}',
            'sold': f'Sold_{i}', 'totalconsumed': f'Total Consumed_{i}'
        })
        Uppsala[i]['Date'] = pd.to_datetime(Uppsala[i]['DATE_TIME'])
        Uppsala[i].drop(columns=['Unnamed: 0', 'clientid', 'DATE_TIME'], errors='ignore', inplace=True)

    for j, df in enumerate(Halmstad):
        Halmstad[j] = df.rename(columns={
            'bought': f'Bought_{j}', 'produced': f'Produced_{j}',
            'sold': f'Sold_{j}', 'totalconsumed': f'Total Consumed_{j}'
        })
        Halmstad[j]['Date'] = pd.to_datetime(Halmstad[j]['DATE_TIME'])
        Halmstad[j].drop(columns=['Unnamed: 0', 'clientid', 'DATE_TIME'], errors='ignore', inplace=True)

    return Uppsala, Halmstad


def load_weather_data(dataset_dir: str):
    """Load weather and radiation CSVs from Dataset/weather/."""
    w_dir = os.path.join(dataset_dir, 'weather')

    w_u = pd.read_csv(os.path.join(w_dir, 'Upsalla_Weather.csv')).iloc[:, [0,1,2,3,4,6,7,8,9]]
    w_h = pd.read_csv(os.path.join(w_dir, 'Halmstad_Weather.csv')).iloc[:, [0,1,2,3,4,6,7,8,9]]
    W_u = w_u.rename(columns={'time': 'Date'})
    W_h = w_h.rename(columns={'time': 'Date'})
    W_u['Date'] = pd.to_datetime(W_u['Date'])
    W_h['Date'] = pd.to_datetime(W_h['Date'])

    rad_u = pd.read_csv(os.path.join(w_dir, 'Upsalla_rad.csv'))
    rad_h = pd.read_csv(os.path.join(w_dir, 'Halmstad_rad.csv'))
    R_u = rad_u.rename(columns={'value': 'Radiation', 'date_time': 'Date'})
    R_h = rad_h.rename(columns={'value': 'Radiation', 'date_time': 'Date'})
    R_u['Date'] = pd.to_datetime(R_u['Date']).dt.tz_localize(None)
    R_h['Date'] = pd.to_datetime(R_h['Date']).dt.tz_localize(None)

    return W_u, W_h, R_u, R_h


def merge_and_save_combined(dataset_dir: str):
    """Merge prosumer lists with weather and save combined CSVs.

    Writes:
        Dataset/combined_datasets/Uppsala_combined.csv
        Dataset/combined_datasets/Halmstad_combined.csv
    """
    Uppsala, Halmstad = load_prosumer_data(dataset_dir)
    W_u, W_h, R_u, R_h = load_weather_data(dataset_dir)

    merged_u = reduce(lambda x, y: x.merge(y, on='Date', how='inner'), Uppsala)
    merged_h = reduce(lambda x, y: x.merge(y, on='Date', how='inner'), Halmstad)

    merged_u = merged_u.merge(W_u, on='Date', how='inner').merge(R_u, on='Date', how='inner')
    merged_h = merged_h.merge(W_h, on='Date', how='inner').merge(R_h, on='Date', how='inner')

    out_dir = os.path.join(dataset_dir, 'combined_datasets')
    os.makedirs(out_dir, exist_ok=True)
    merged_u.to_csv(os.path.join(out_dir, 'Uppsala_combined.csv'), index=False)
    merged_h.to_csv(os.path.join(out_dir, 'Halmstad_combined.csv'), index=False)
    print(f'Saved combined datasets to {out_dir}')
    return merged_u, merged_h


def build_prosumer_datasets(dataset_dir: str):
    """Split the wide combined CSVs into one file per prosumer per target.

    Reads Uppsala_combined.csv and Halmstad_combined.csv (written by
    merge_and_save_combined), then for each prosumer index i extracts its
    four columns (Bought_i, Produced_i, Sold_i, Total Consumed_i) together
    with all weather / radiation columns (no suffix) and the Date column,
    renames them to the canonical unsuffixed names, and writes:
        {City}_Consumption_{i}.csv  – all hours, target = 'Total Consumed'
        {City}_Production_{i}.csv   – daylight hours only, target = 'Produced'

    This produces the per-prosumer files that preprocess_scale_and_create_multistep
    and the Conference Paper pipeline expect.
    """
    comb_dir = os.path.join(dataset_dir, 'combined_datasets')

    prosumer_col_prefixes = ('Bought_', 'Produced_', 'Sold_', 'Total Consumed_')

    for city, n_prosumers in [('Uppsala', 7), ('Halmstad', 5)]:
        fpath = os.path.join(comb_dir, f'{city}_combined.csv')
        if not os.path.exists(fpath):
            print(f'[skip] {fpath} not found — run merge_and_save_combined first')
            continue

        df = pd.read_csv(fpath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Columns that belong to prosumers have a numeric suffix
        weather_cols = [
            c for c in df.columns
            if not any(c.startswith(p) for p in prosumer_col_prefixes)
            and c != 'Date'
        ]

        for i in range(n_prosumers):
            suffix = f'_{i}'
            prosumer_specific = {
                f'Bought{suffix}':         'Bought',
                f'Produced{suffix}':       'Produced',
                f'Sold{suffix}':           'Sold',
                f'Total Consumed{suffix}': 'Total Consumed',
            }
            # Keep only columns that actually exist (guard against missing data)
            available = {k: v for k, v in prosumer_specific.items()
                         if k in df.columns}
            if not available:
                print(f'[skip] {city} prosumer {i}: expected columns not found')
                continue

            cols_to_keep = ['Date'] + list(available.keys()) + weather_cols
            df_p = df[cols_to_keep].rename(columns=available).copy()

            df_p.to_csv(
                os.path.join(comb_dir, f'{city}_Consumption_{i}.csv'),
                index=False,
            )
            df_night_filtered = apply_nighttime_filter(df_p)
            df_night_filtered.to_csv(
                os.path.join(comb_dir, f'{city}_Production_{i}.csv'),
                index=False,
            )
            print(f'Saved {city} prosumer {i}  '
                  f'({len(df_p)} consumption / '
                  f'{len(df_night_filtered)} production rows)')


# ---------------------------------------------------------------------------
# Preprocessing, scaling, and multi-step window creation
# ---------------------------------------------------------------------------

def _preprocess_and_scale(df: pd.DataFrame, target_column: str, train_ratio: float = 0.8):
    """Add time features, split 80/20, scale features, return arrays."""
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df.ffill(inplace=True)
    df['Year']  = df.index.year
    df['Month'] = df.index.month
    df['Day']   = df.index.day
    df['Hour']  = df.index.hour

    n_train = int(len(df) * train_ratio)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test  = test.drop(columns=[target_column])
    y_test  = test[target_column]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled, y_train.values, y_test.values


def _create_windows(X_scaled: np.ndarray, y: np.ndarray,
                    input_steps: int, forecast_steps: int):
    """Slide a window over (X, y) to create multi-step samples.

    Returns
    -------
    X_out      : shape (n_windows, input_steps, n_features)
    y_out      : shape (n_windows, forecast_steps)
    y_last_obs : shape (n_windows,) — last observed target value before each window
    """
    X_out, y_out, y_last_obs = [], [], []
    n = len(X_scaled) - input_steps - forecast_steps + 1
    for i in range(n):
        X_out.append(X_scaled[i: i + input_steps])
        y_out.append(y[i + input_steps: i + input_steps + forecast_steps])
        y_last_obs.append(y[i + input_steps - 1])
    return np.array(X_out), np.array(y_out), np.array(y_last_obs)


def preprocess_scale_and_create_multistep(
    dataset_dir: str,
    input_steps: int = 24,
    forecast_steps: int = 48,
    train_ratio: float = 0.8,
) -> dict:
    """Load all prosumer CSVs, scale, and create multi-step windows.

    Returns a dict with keys:
        'halmstad_consumption', 'halmstad_production',
        'uppsala_consumption',  'uppsala_production'
    Each value is a list of (X_train, y_train, X_test, y_test) tuples.
    """
    comb_dir = os.path.join(dataset_dir, 'combined_datasets')
    result = {
        'halmstad_consumption': [],
        'halmstad_production':  [],
        'uppsala_consumption':  [],
        'uppsala_production':   [],
    }
    target_map = {
        'consumption': 'Total Consumed',
        'production':  'Produced',
    }

    for fname in sorted(os.listdir(comb_dir)):
        if not fname.endswith('.csv'):
            continue
        city, mode = None, None
        for c in ('Halmstad', 'Uppsala'):
            for m in ('Consumption', 'Production'):
                if f'{c}_{m}' in fname:
                    city, mode = c.lower(), m.lower()
        if city is None:
            continue

        df = pd.read_csv(os.path.join(comb_dir, fname))
        target = target_map[mode]
        if target not in df.columns:
            continue

        scaler, Xtr, Xte, ytr, yte = _preprocess_and_scale(df, target, train_ratio)
        Xtr_w, ytr_w, _          = _create_windows(Xtr, ytr, input_steps, forecast_steps)
        Xte_w, yte_w, yte_last   = _create_windows(Xte, yte, input_steps, forecast_steps)
        result[f'{city}_{mode}'].append((Xtr_w, ytr_w, Xte_w, yte_w, yte_last))

    for k, v in result.items():
        print(f'{k}: {len(v)} prosumers loaded')

    return result
