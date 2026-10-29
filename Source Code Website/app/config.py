"""
Konfigurasi TFT sesuai ID8_run3_LR0,0005,BS_32vers2.ipynb
"""

# TimeSeriesDataSet Configuration
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 7
MIN_ENCODER_LENGTH = 30  # max_encoder_length // 2
MIN_PREDICTION_LENGTH = 1

# Target column
TARGET = "close"
GROUP_IDS = ["group_id"]

# Static categoricals
STATIC_CATEGORICALS = ["group_id"]

# Time varying known categoricals
TIME_VARYING_KNOWN_CATEGORICALS = ["day_of_week", "month"]

# Time varying known reals
TIME_VARYING_KNOWN_REALS = ["time_idx", "year_num"]

# Missing flags and observed unknown reals
MISSING_FLAGS = [
    "usd_open_missing", "usd_close_missing", "gspc_open_missing",
    "gspc_close_missing", "oil_dcoilwtico_missing", "fed_dff_pct_missing"
]

UNKNOWN_REALS = [
    "open", "close",
    "close_lag1", "close_lag5", "close_ma7", "close_ma20", "close_std7",
    "log_close", "log_close_lag1", "log_close_ma7", "close_pct_from_ma20",
    "gold_ret1", "gold_ret5", "gold_vol5", "gold_vol20",
    "usd_open", "usd_close", "usd_close_lag1", "usd_ret1",
    "gspc_open", "gspc_close", "gspc_close_lag1", "gspc_ret1",
    "oil_dcoilwtico", "oil_lag1", "oil_ret1", "fed_dff_pct",
] + MISSING_FLAGS

DEFAULT_VALUES = {
    "open": 2000.0,
    "close": 2000.0,
    "tickvol": 0,
    "close_lag1": 2000.0,
    "close_lag5": 2000.0,
    "close_ma7": 2000.0,
    "close_ma20": 2000.0,
    "close_std7": 20.0,
    "log_close": 7.60,
    "log_close_lag1": 7.60,
    "log_close_ma7": 7.60,
    "close_pct_from_ma20": 0.0,
    "gold_ret1": 0.0,
    "gold_ret5": 0.0,
    "gold_vol5": 0.01,
    "gold_vol20": 0.01,
    "usd_open": 104.0,
    "usd_close": 104.0,
    "usd_close_lag1": 104.0,
    "usd_ret1": 0.0,
    "gspc_open": 5000.0,
    "gspc_close": 5000.0,
    "gspc_close_lag1": 5000.0,
    "gspc_ret1": 0.0,
    "oil_dcoilwtico": 75.0,
    "oil_lag1": 75.0,
    "oil_ret1": 0.0,
    "fed_dff_pct": 5.33,
    "usd_open_missing": 0.0,
    "usd_close_missing": 0.0,
    "gspc_open_missing": 0.0,
    "gspc_close_missing": 0.0,
    "oil_dcoilwtico_missing": 0.0,
    "fed_dff_pct_missing": 0.0,
}

API_COLUMN_MAPPING = {
    "close": "close",
    "open": "open",
    "usd_close": "usd_close",
    "usd_open": "usd_open",
    "gspc_close": "gspc_close",
    "gspc_open": "gspc_open",
    "oil_dcoilwtico": "oil_dcoilwtico",
    "fed_dff_pct": "fed_dff_pct",
}
