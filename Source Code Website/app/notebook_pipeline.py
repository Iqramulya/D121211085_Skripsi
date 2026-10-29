"""Preprocessing from ID8_run3_LR0,0005,BS_32vers2.ipynb, cells 10 and 12."""
import numpy as np
import pandas as pd

from .config import UNKNOWN_REALS as MODEL_REALS

RAW_COLUMNS = ['date', 'open', 'close', 'dxy_open', 'dxy_close',
               'gspc_open', 'gspc_close', 'oil_dcoilwtico', 'fed_dff_pct']
NOTEBOOK_NAME = 'ID8_run3_LR0,0005,BS_32vers2.ipynb'


def prepare_history(history):
    df = pd.DataFrame(history).copy()
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError('Kolom training notebook belum lengkap: ' + ', '.join(sorted(missing)))
    dates = pd.to_datetime(df['date'], errors='raise')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Tanggal CSV kosong atau duplikat.')
    for col in set(RAW_COLUMNS[1:]) | (set(df.columns) - {'date'}):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if not np.isfinite(df['close']).all() or (df['close'] <= 0).any():
        raise ValueError('Harga emas harus positif.')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Cell 10 uses the original USD series before cell 12 replaces it with DXY.
    # Older scenario histories can have an entirely empty legacy USD column.
    # Treat them like DXY-only CSVs; retain the legacy series when observed so
    # the full training dataset continues to match the notebook exactly.
    if 'usd_close' not in df or not np.isfinite(df['usd_close']).any():
        df['usd_close'] = df['dxy_close']

    # Fill missing values
    ffill_cols = [
        'oil_dcoilwtico', 'fed_rate_pct', 'fed_target_rate_from_pct',
        'fed_target_rate_to_pct', 'cpi_cpiaucsl',
        'usd_open', 'usd_close', 'usd_adj_close',
        'gspc_open', 'gspc_close',
    ]
    for col in ffill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Feature Engineering
    df['close_lag1']  = df['close'].shift(1)
    df['close_lag5']  = df['close'].shift(5)
    df['close_ma7']   = df['close'].rolling(7).mean()
    df['close_ma20']  = df['close'].rolling(20).mean()
    df['close_std7']  = df['close'].rolling(7).std()

    df['log_close']   = np.log(df['close'])
    df['log_close_lag1'] = df['log_close'].shift(1)
    df['log_close_ma7']  = df['log_close'].rolling(7).mean()

    df['close_pct_from_ma20'] = (df['close'] - df['close_ma20']) / df['close_ma20']

    df['gold_ret1']   = df['close'].pct_change()
    df['gold_ret5']   = df['close'].pct_change(5)
    df['usd_ret1']    = df['usd_close'].pct_change()
    df['gspc_ret1']   = df['gspc_close'].pct_change()
    df['oil_ret1']    = df['oil_dcoilwtico'].pct_change()

    df['usd_close_lag1']  = df['usd_close'].shift(1)
    df['oil_lag1']        = df['oil_dcoilwtico'].shift(1)
    df['gspc_close_lag1'] = df['gspc_close'].shift(1)

    df['gold_vol5']   = df['gold_ret1'].rolling(5).std()
    df['gold_vol20']  = df['gold_ret1'].rolling(20).std()

    # Drop NaN
    required_cols = [
        'close_lag1', 'close_lag5', 'close_ma7', 'close_ma20', 'close_std7',
        'log_close', 'log_close_lag1', 'log_close_ma7', 'close_pct_from_ma20',
        'gold_ret1', 'gold_ret5', 'usd_ret1', 'gspc_ret1',  'oil_ret1',
        'usd_close_lag1', 'oil_lag1', 'gspc_close_lag1',
        'gold_vol5', 'gold_vol20',
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Metadata
    df['time_idx']    = range(len(df))
    df['group_id']    = 'XAU_USD'
    df['day_of_week'] = df['date'].dt.day_name().astype(str)
    df['month']       = df['date'].dt.month.astype(str)
    df['year_num']    = (df['date'].dt.year - 2021).astype(float)

    # Cell 12: DXY, limited forward fill, training-only medians, missing flags.
    train_mask = df['time_idx'] < int(len(df) * 0.80)
    df['usd_open'] = df['dxy_open']
    df['usd_close'] = df['dxy_close']
    base_features = ['usd_open', 'usd_close', 'gspc_open', 'gspc_close',
                     'oil_dcoilwtico', 'fed_dff_pct']
    for col in base_features:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        df[f'{col}_missing'] = values.isna().astype(float)
        historical = values.ffill(limit=5)
        fallback = historical.loc[train_mask].median()
        if not np.isfinite(fallback):
            raise ValueError(f'{col}: tidak ada nilai valid dalam training.')
        df[col] = historical.fillna(fallback)
    for source, lag, ret in [('usd_close', 'usd_close_lag1', 'usd_ret1'),
                             ('gspc_close', 'gspc_close_lag1', 'gspc_ret1'),
                             ('oil_dcoilwtico', 'oil_lag1', 'oil_ret1')]:
        df[lag] = df[source].shift(1)
        df[ret] = df[source].pct_change(fill_method=None)
    for col in MODEL_REALS:
        if col == 'close':
            continue
        values = df[col].replace([np.inf, -np.inf], np.nan)
        fallback = values.loc[train_mask].median()
        if not np.isfinite(fallback):
            raise ValueError(f'{col}: seluruh nilai training kosong/inf.')
        df[col] = values.fillna(fallback)

    return df


def build_notebook_forecast(history):
    df = prepare_history(history)
    if len(df) < 60:
        raise ValueError('Dibutuhkan minimal 80 observasi mentah: 20 untuk fitur dan 60 untuk encoder TFT.')
    if not np.isfinite(df[MODEL_REALS].to_numpy(dtype=float)).all():
        raise ValueError('Fitur training kosong atau tidak valid; gunakan data historis lengkap tanpa nilai pengganti sintetis.')
    last_data = df.tail(60).copy()
    future_dates = pd.bdate_range(df['date'].max() + pd.Timedelta(days=1), periods=7)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['group_id'] = 'XAU_USD'
    future_df['day_of_week'] = future_df['date'].dt.day_name().astype(str)
    future_df['month'] = future_df['date'].dt.month.astype(str)
    future_df['year_num'] = (future_df['date'].dt.year - 2021).astype(float)
    future_df['time_idx'] = list(range(int(df['time_idx'].max()) + 1, int(df['time_idx'].max()) + 8))
    for col in df.columns:
        if col not in future_df.columns:
            future_df[col] = df[col].iloc[-1]
    return pd.concat([last_data, future_df], ignore_index=True)
