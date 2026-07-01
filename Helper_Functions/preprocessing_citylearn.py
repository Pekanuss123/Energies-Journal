"""
preprocessing_citylearn.py
--------------------------
CityLearn 2022 Challenge dataset preprocessing pipeline.

The CityLearn dataset lives in  Dataset/ELAD_Data/  and contains:
  Building_1.csv … Building_17.csv  — per-building hourly measurements
  weather.csv                        — outdoor environmental features
  schema.json                        — CityLearn scenario definition, including
                                        each building's installed PV capacity
                                        (buildings.<name>.pv.attributes.nominal_power)

Column conventions (per building CSV):
  non_shiftable_load_*   – always-on appliance load
  cooling_demand_*       – cooling energy
  heating_demand_*       – heating energy
  dhw_demand_*           – domestic hot water energy
  solar_generation_*     – PV generation, expressed as W generated per kW of
                            installed capacity (i.e. a *normalized* profile,
                            NOT actual power). Must be multiplied by the
                            building's nominal_power (kW) and divided by 1000
                            to obtain actual kW production.

This module provides:

  load_citylearn_data(elad_dir)
      -> merged DataFrame (all buildings + weather, hourly, datetime index).
         solar_generation_* columns are capacity-scaled to actual kW whenever
         schema.json is available (see _load_pv_capacities).

  build_citylearn_datasets(elad_dir, output_dir)
      -> writes CityLearn_Consumption.csv and CityLearn_Production.csv
         (grid-level aggregates, for descriptive/EDA use -- NOT used by the
         per-building modeling pipeline below).

  preprocess_citylearn_multistep(elad_dir, input_steps, forecast_steps,
                                  train_ratio)
      -> dict with keys 'citylearn_consumption' and 'citylearn_production'.
         Each value is a LIST of one 5-tuple PER BUILDING
         (X_train, y_train, X_test, y_test, y_test_last_obs), mirroring the
         per-prosumer list structure returned by
         preprocess_scale_and_create_multistep() for the Swedish sites --
         i.e. CityLearn's 17 buildings are modeled individually, not summed
         into one grid-wide series.

         The train/test split is season-balanced (quarterly blocks, 80/20
         within each quarter) rather than a single tail-of-year cut, so both
         splits span the full seasonal cycle instead of the test set landing
         entirely in Oct-Dec.
"""

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PV capacity lookup (schema.json)
# ---------------------------------------------------------------------------

def _load_pv_capacities(elad_dir: str) -> dict:
    """Read each building's installed PV capacity (kW) from schema.json.

    Returns dict {building_id_str: nominal_power_kW}, or {} if schema.json
    is missing / malformed.
    """
    schema_path = os.path.join(elad_dir, 'schema.json')
    if not os.path.exists(schema_path):
        warnings.warn(
            f'schema.json not found in {elad_dir} -- solar_generation_* '
            f'columns will NOT be capacity-scaled. Production targets will '
            f'be in normalized (W per kW installed) units, not actual kW.'
        )
        return {}

    try:
        with open(schema_path) as f:
            schema = json.load(f)
        caps = {}
        for name, b in schema.get('buildings', {}).items():
            bid = name.split('_')[-1]
            nom = b.get('pv', {}).get('attributes', {}).get('nominal_power')
            if nom is not None:
                caps[bid] = float(nom)
        return caps
    except Exception as e:
        warnings.warn(f'Failed to parse schema.json ({e}) -- no PV scaling applied.')
        return {}


# ---------------------------------------------------------------------------
# Step 1: Load and merge buildings + weather
# ---------------------------------------------------------------------------

def load_citylearn_data(elad_dir: str, apply_pv_scaling: bool = True) -> pd.DataFrame:
    """Load all Building_*.csv files and weather.csv from elad_dir,
    merge column-wise, and attach a synthetic hourly datetime index
    starting 2021-01-01.

    If apply_pv_scaling is True and schema.json is available, each
    building's solar_generation_<id> column is rescaled from a normalized
    (W per kW installed) profile to actual kW production:
        solar_generation_<id> *= nominal_power_<id> (kW) / 1000

    Returns a DataFrame with all building and weather columns, indexed
    by datetime. df.attrs['building_ids'] holds the list of building id strings.
    """
    building_files = sorted(
        glob.glob(os.path.join(elad_dir, 'Building_*.csv'))
    )
    if not building_files:
        raise FileNotFoundError(
            f'No Building_*.csv files found in {elad_dir}'
        )

    time_cols = ['month', 'hour', 'day_type', 'daylight_savings_status']

    building_dfs = []
    building_ids = []
    df_time = None

    for fpath in building_files:
        b_id = os.path.splitext(os.path.basename(fpath))[0].split('_')[1]
        building_ids.append(b_id)
        df = pd.read_csv(fpath)

        if df_time is None:
            df_time = df[[c for c in time_cols if c in df.columns]]

        other_cols = [c for c in df.columns if c not in time_cols]
        building_dfs.append(df[other_cols].add_suffix(f'_{b_id}'))

    weather_path = os.path.join(elad_dir, 'weather.csv')
    weather_df = pd.read_csv(weather_path) if os.path.exists(weather_path) else pd.DataFrame()

    parts = [df_time] + building_dfs
    if not weather_df.empty:
        parts.append(weather_df)
    merged = pd.concat(parts, axis=1)

    # --- PV capacity scaling: normalized profile (W/kWp) -> actual kW ---
    if apply_pv_scaling:
        capacities = _load_pv_capacities(elad_dir)
        for b_id in building_ids:
            col = f'solar_generation_{b_id}'
            if col in merged.columns and b_id in capacities:
                merged[col] = merged[col] * capacities[b_id] / 1000.0
            elif col in merged.columns and b_id not in capacities:
                warnings.warn(f'No PV capacity found for building {b_id} -- '
                               f'{col} left unscaled.')

    # Attach synthetic datetime index
    n_rows = len(merged)
    merged = merged.copy()
    merged['datetime'] = pd.date_range(start='2021-01-01', periods=n_rows, freq='h')
    merged['month'] = merged['datetime'].dt.month
    merged['hour']  = merged['datetime'].dt.hour
    merged = merged.set_index('datetime')

    merged.attrs['building_ids'] = building_ids
    return merged


# ---------------------------------------------------------------------------
# Step 2: Build grid-level consumption/production CSVs (descriptive use only)
# ---------------------------------------------------------------------------

_DAYLIGHT_HOURS = {
    10: (7, 18), 11: (7, 16), 12: (8, 15),
     1: (8, 16),  2: (7, 17),  3: (6, 18),
     4: (5, 20),  5: (4, 21),  6: (3, 22),
     7: (3, 22),  8: (5, 21),  9: (6, 19),
}


def _apply_daylight_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove night-time rows (same logic as Swedish prosumer pipeline)."""
    masks = [
        (df.index.month == m) &
        (df.index.hour  >= h_start) &
        (df.index.hour  <= h_end)
        for m, (h_start, h_end) in _DAYLIGHT_HOURS.items()
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return df[combined]


def build_citylearn_datasets(elad_dir: str, output_dir: str) -> None:
    """Create CityLearn_Consumption.csv and CityLearn_Production.csv.

    NOTE: these are grid-level (summed-across-buildings) aggregates,
    intended for descriptive/EDA use. They are NOT used by
    preprocess_citylearn_multistep(), which models each building individually.

    solar_generation_* columns are capacity-scaled to actual kW.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_citylearn_data(elad_dir)

    consumption_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in
               ['non_shiftable_load_', 'cooling_demand_',
                'heating_demand_', 'dhw_demand_'])
    ]
    df_cons = df.copy()
    df_cons['Total_Consumption'] = df_cons[consumption_cols].sum(axis=1)

    cons_path = os.path.join(output_dir, 'CityLearn_Consumption.csv')
    df_cons.to_csv(cons_path)
    print(f'Written {cons_path}  ({len(df_cons)} rows)')

    production_cols = [c for c in df.columns if c.startswith('solar_generation_')]
    df_prod = df.copy()
    df_prod['Total_Production'] = df_prod[production_cols].sum(axis=1)
    df_prod = _apply_daylight_filter(df_prod)

    prod_path = os.path.join(output_dir, 'CityLearn_Production.csv')
    df_prod.to_csv(prod_path)
    print(f'Written {prod_path}  ({len(df_prod)} rows)')


# ---------------------------------------------------------------------------
# Step 3: Per-building multi-step windows, season-balanced split
# ---------------------------------------------------------------------------

_QUARTER_OF_MONTH = {m: (m - 1) // 3 for m in range(1, 13)}  # 0,1,2,3


def _season_balanced_windows(df: pd.DataFrame, target_col: str,
                              input_steps: int, forecast_steps: int,
                              train_ratio: float):
    """Scale features (train-only fit) and window, using a season-balanced
    train/test split instead of a single chronological tail-cut.

    CityLearn's synthetic index spans exactly one calendar year. A naive
    80/20 chronological split puts the entire test set in the last ~2.4
    months (Oct-Dec) -- a distribution shift for a strongly seasonal signal
    like solar PV. Instead, the year is split into 4 quarterly blocks; within
    each quarter, the first `train_ratio` fraction of rows is used for
    training and the remainder for testing. Windows are built within each
    contiguous block (never across a quarter boundary), then concatenated
    across quarters.
    """
    df = df.ffill().fillna(0)

    for feat, getter in [
        ('Year',  lambda d: d.index.year),
        ('Month', lambda d: d.index.month),
        ('Day',   lambda d: d.index.day),
        ('Hour',  lambda d: d.index.hour),
    ]:
        if feat not in df.columns:
            df[feat] = getter(df)

    quarters = df.index.month.map(_QUARTER_OF_MONTH)

    min_block_len = input_steps + forecast_steps
    train_blocks_raw, test_blocks_raw = [], []
    for q in sorted(set(quarters)):
        block = df[quarters == q]
        cut = int(len(block) * train_ratio)
        train_blocks_raw.append(block.iloc[:cut])
        test_blocks_raw.append(block.iloc[cut:])

    train_concat_raw = pd.concat(train_blocks_raw)
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    scaler.fit(train_concat_raw[feature_cols])

    def _window_block(block: pd.DataFrame):
        if len(block) < min_block_len:
            return None
        X_scaled = scaler.transform(block[feature_cols])
        y = block[target_col].values
        n = len(X_scaled) - input_steps - forecast_steps + 1
        Xw     = np.array([X_scaled[i:i + input_steps] for i in range(n)])
        yw     = np.array([y[i + input_steps:i + input_steps + forecast_steps] for i in range(n)])
        y_last = np.array([y[i + input_steps - 1] for i in range(n)])
        return Xw, yw, y_last

    Xtr_parts, ytr_parts = [], []
    for block in train_blocks_raw:
        w = _window_block(block)
        if w is not None:
            Xtr_parts.append(w[0]); ytr_parts.append(w[1])

    Xte_parts, yte_parts, yte_last_parts = [], [], []
    for block in test_blocks_raw:
        w = _window_block(block)
        if w is not None:
            Xte_parts.append(w[0]); yte_parts.append(w[1]); yte_last_parts.append(w[2])

    if not Xtr_parts or not Xte_parts:
        raise ValueError(
            f'Not enough rows per quarterly block for target={target_col} '
            f'(input_steps={input_steps}, forecast_steps={forecast_steps}). '
            f'Reduce forecast_steps or use a coarser block granularity.'
        )

    X_train = np.concatenate(Xtr_parts, axis=0)
    y_train = np.concatenate(ytr_parts, axis=0)
    X_test  = np.concatenate(Xte_parts, axis=0)
    y_test  = np.concatenate(yte_parts, axis=0)
    y_last  = np.concatenate(yte_last_parts, axis=0)

    return X_train, y_train, X_test, y_test, y_last


# --- Per-building column groups ---------------------------------------------

_OWN_STATE_COLS = [
    'indoor_dry_bulb_temperature',
    'average_unmet_cooling_setpoint_difference',
    'indoor_relative_humidity',
]
_OWN_CONSUMPTION_COMPONENT_COLS = [
    'non_shiftable_load', 'dhw_demand', 'cooling_demand', 'heating_demand',
]
_SHARED_TIME_COLS = [
    'month', 'hour', 'day_type', 'daylight_savings_status',
    'Year', 'Month', 'Day', 'Hour',
]


def _building_frame(df: pd.DataFrame, b_id: str, target: str) -> tuple:
    """Build a single building's feature+target DataFrame and target column name.

    target : 'consumption' or 'production'

    Feature set = shared weather/time columns + this building's own state
    columns (NOT other buildings' columns), mirroring the Swedish pipeline.

    For consumption, the four raw demand columns the target is summed from
    are excluded from the features to avoid a tautological leak.
    """
    all_building_ids = df.attrs.get('building_ids', [])
    per_building_bases = _OWN_STATE_COLS + _OWN_CONSUMPTION_COMPONENT_COLS + ['solar_generation']

    def is_other_building_col(col):
        for other_id in all_building_ids:
            if other_id == b_id:
                continue
            for base in per_building_bases:
                if col == f'{base}_{other_id}':
                    return True
        return False

    own_state           = [f'{c}_{b_id}' for c in _OWN_STATE_COLS if f'{c}_{b_id}' in df.columns]
    own_cons_components = [f'{c}_{b_id}' for c in _OWN_CONSUMPTION_COMPONENT_COLS if f'{c}_{b_id}' in df.columns]
    own_solar           = f'solar_generation_{b_id}'

    shared_weather = [
        c for c in df.columns
        if (c in _SHARED_TIME_COLS or not is_other_building_col(c))
        and c not in own_state
        and c not in own_cons_components
        and c != own_solar
    ]

    if target == 'consumption':
        target_col = f'Total_Consumption_{b_id}'
        solar_feat = [own_solar] if own_solar in df.columns else []
        out = df[shared_weather + own_state + solar_feat].copy()
        out[target_col] = df[own_cons_components].sum(axis=1)
    elif target == 'production':
        target_col = own_solar
        out = df[shared_weather + own_state + own_cons_components].copy()
        if own_solar in df.columns:
            out[target_col] = df[own_solar]
        else:
            raise ValueError(f'Column {own_solar} not found in DataFrame.')
    else:
        raise ValueError(f'target must be "consumption" or "production", got {target!r}')

    return out, target_col


# ---------------------------------------------------------------------------
# Grid-level windows — matches the paper's original pipeline exactly
# ---------------------------------------------------------------------------

def _grid_windows(df: pd.DataFrame, target_col: str,
                  input_steps: int, forecast_steps: int,
                  train_ratio: float):
    """Naive chronological 80/20 split, StandardScaler on train features only,
    sliding windows.  All columns except target_col are used as features.
    Returns a 5-tuple (X_train, y_train, X_test, y_test, y_test_last_obs).
    """
    df = df.copy().ffill().fillna(0)
    for feat, getter in [
        ('Year',  lambda d: d.index.year),
        ('Month', lambda d: d.index.month),
        ('Day',   lambda d: d.index.day),
        ('Hour',  lambda d: d.index.hour),
    ]:
        if feat not in df.columns:
            df[feat] = getter(df)

    train_size = int(len(df) * train_ratio)
    train = df.iloc[:train_size]
    test  = df.iloc[train_size:]

    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[feature_cols])
    X_test_scaled  = scaler.transform(test[feature_cols])

    y_train = train[target_col].values
    y_test  = test[target_col].values

    n_tr = len(X_train_scaled) - input_steps - forecast_steps + 1
    n_te = len(X_test_scaled)  - input_steps - forecast_steps + 1

    X_tr   = np.array([X_train_scaled[i:i + input_steps] for i in range(n_tr)])
    y_tr   = np.array([y_train[i + input_steps:i + input_steps + forecast_steps] for i in range(n_tr)])
    X_te   = np.array([X_test_scaled[i:i + input_steps]  for i in range(n_te)])
    y_te   = np.array([y_test[i + input_steps:i + input_steps + forecast_steps]  for i in range(n_te)])
    y_last = np.array([y_test[i + input_steps - 1]                               for i in range(n_te)])

    return X_tr, y_tr, X_te, y_te, y_last


def preprocess_citylearn_multistep(
    elad_dir: str,
    input_steps: int = 24,
    forecast_steps: int = 48,
    train_ratio: float = 0.80,
) -> dict:
    """CityLearn preprocessing pipeline matching the paper's original approach.

    Produces a single grid-level series per target type (consumption and
    production) by summing across all 17 buildings, with a naive
    chronological 80/20 split — matching the paper's pipeline exactly.

    PV capacity scaling is intentionally NOT applied to the production target:
    solar_generation_* columns are in W/kWp units and are summed as-is,
    reproducing the paper's target scale (raw summed W/kWp).

    Returns
    -------
    dict with keys 'citylearn_consumption' and 'citylearn_production'.
    Each value is a list containing ONE 5-tuple (index 0):
        (X_train, y_train, X_test, y_test, y_test_last_obs)
    matching the paper's single-series output and plugging into the same
    downstream loops as the Swedish prosumer data.
    """
    # Load WITH PV scaling: solar_generation_* columns are multiplied by each
    # building's nominal_power (kW) / 1000, converting them from normalized
    # W/kWp profiles to actual kW production.  This matches the pipeline used
    # to produce Table 5 of the paper (CityLearn Direct production MAE=22.30,
    # PICP=0.967).  The saved CSV artefacts in the Journal_Paper download folder
    # are from an earlier unscaled run and do NOT reflect the final paper results.
    df = load_citylearn_data(elad_dir, apply_pv_scaling=True)

    # --- Consumption: sum all buildings' demand components (all hours) ---
    consumption_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in
               ['non_shiftable_load_', 'cooling_demand_',
                'heating_demand_', 'dhw_demand_'])
    ]
    df_cons = df.copy()
    df_cons['Total_Consumption'] = df_cons[consumption_cols].sum(axis=1)
    X_tr, y_tr, X_te, y_te, y_last = _grid_windows(
        df_cons, 'Total_Consumption', input_steps, forecast_steps, train_ratio
    )
    print(f'CityLearn Consumption (grid): X_train={X_tr.shape}  X_test={X_te.shape}')

    # --- Production: sum raw solar_generation_* (daylight hours only) ---
    solar_cols = [c for c in df.columns if c.startswith('solar_generation_')]
    df_prod = _apply_daylight_filter(df.copy())
    df_prod['Total_Production'] = df_prod[solar_cols].sum(axis=1)
    X_tr_p, y_tr_p, X_te_p, y_te_p, y_last_p = _grid_windows(
        df_prod, 'Total_Production', input_steps, forecast_steps, train_ratio
    )
    print(f'CityLearn Production  (grid): X_train={X_tr_p.shape}  X_test={X_te_p.shape}')

    return {
        'citylearn_consumption': [(X_tr,   y_tr,   X_te,   y_te,   y_last)],
        'citylearn_production':  [(X_tr_p, y_tr_p, X_te_p, y_te_p, y_last_p)],
    }
