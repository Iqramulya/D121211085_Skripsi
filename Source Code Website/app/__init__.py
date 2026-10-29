import os
import glob
import logging
import hashlib
import json
from typing import Dict

import pandas as pd
import numpy as np

from .config import (
    UNKNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    MIN_ENCODER_LENGTH,
    MIN_PREDICTION_LENGTH,
    TARGET,
    GROUP_IDS,
    STATIC_CATEGORICALS,
    TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS,
    DEFAULT_VALUES,
)

_LOADED_MODELS: Dict[str, object] = {}
# Set default model path to models/ folder in project directory
# From app/__init__.py, we need to go: app/ -> flask_tft_app/ -> models
_DEFAULT_LOGS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "training_v2"))
_BASE_LOGS = os.environ.get("TFT_LIGHTNING_LOGS", os.environ.get("TFT_MODEL_BASE", _DEFAULT_LOGS))


def discover_versions(base_path: str = None):
    base = base_path or _BASE_LOGS
    versions = []
    if not os.path.isdir(base):
        return versions
    for path in sorted(glob.glob(os.path.join(base, "version_*"))):
        versions.append(os.path.basename(path))
    return versions


def best_ckpt_for_version(version: str, base_path: str = None):
    base = base_path or _BASE_LOGS
    version_dir = os.path.join(base, version)
    ckpt_dir = os.path.join(version_dir, "checkpoints")

    logging.info(f"Looking for checkpoints in: {ckpt_dir}")

    manifest_path = os.path.join(version_dir, 'source.json')
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as source:
                manifest = json.load(source)
            filename = os.path.basename(manifest['member'])
            selected = os.path.join(ckpt_dir, filename)
            if filename.endswith('.ckpt') and os.path.isfile(selected):
                return selected
            # The previous installer renamed its exact weights to best_tft_model.
            # Accept that legacy name only when its content matches the manifest.
            renamed = os.path.join(ckpt_dir, 'best_tft_model.ckpt')
            if manifest.get('sha256') and os.path.isfile(renamed):
                with open(renamed, 'rb') as weights:
                    digest = hashlib.file_digest(weights, 'sha256').hexdigest()
                if digest == manifest['sha256']:
                    return renamed
            logging.error('Checkpoint declared in %s is missing', manifest_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logging.error('Invalid checkpoint manifest %s: %s', manifest_path, exc)
        return None

    if os.path.isdir(ckpt_dir):
        # Priority 1: CPU-trained model from model-lama (tft_cpu_model)
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "tft_cpu_model*.ckpt"))):
            if not ckpt.endswith(".bak") and not ckpt.endswith(".bak_new"):
                logging.info(f"Found CPU model checkpoint: {ckpt} for {version}")
                return ckpt
        # Priority 2: GPU training checkpoint from model-lama (epoch=11-step=192.ckpt)
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "epoch=11*.ckpt"))):
            if not ckpt.endswith(".bak") and not ckpt.endswith(".bak_new"):
                logging.info(f"Found GPU training checkpoint: {ckpt} for {version}")
                return ckpt
        # Priority 3: exact best_tft_model.ckpt
        best_exact = os.path.join(ckpt_dir, "best_tft_model.ckpt")
        if os.path.isfile(best_exact):
            logging.info(f"Found best model checkpoint: {best_exact} for {version}")
            return best_exact
        # Priority 4: best_*.ckpt
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "best_*.ckpt"))):
            if not ckpt.endswith(".bak") and not ckpt.endswith(".bak_new"):
                logging.info(f"Found best checkpoint: {ckpt} for {version}")
                return ckpt
        # Priority 5: any non-backup checkpoint
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt"))):
            if not ckpt.endswith(".bak") and not ckpt.endswith(".bak_new"):
                logging.info(f"Found checkpoint (any): {ckpt} for {version}")
                return ckpt
    else:
        logging.warning(f"Checkpoint directory not found: {ckpt_dir}")

    logging.error(f"No checkpoint found for {version}")
    return None


def load_model(version: str, base_path: str = None):
    global _LOADED_MODELS
    ckpt_path = best_ckpt_for_version(version, base_path or _BASE_LOGS)
    cache_key = os.path.abspath(ckpt_path) if ckpt_path else None
    if cache_key in _LOADED_MODELS:
        logging.info(f"Model for {version} already loaded in memory")
        return _LOADED_MODELS[cache_key]
    if not ckpt_path:
        logging.warning(f"No checkpoint found for {version}, cannot load model.")
        return None

    try:
        from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import EncoderNormalizer
        import torch
        import pickle

        logging.info(f"Loading TFT model from: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Use the preprocessing metadata saved with these exact weights.
        # The external training_dataset.pkl was rebuilt with incompatible scalers.
        hparams = checkpoint.get('hyper_parameters', {}).copy()
        parameters = hparams.get('dataset_parameters')
        if not parameters:
            raise ValueError('Checkpoint tidak menyimpan parameter preprocessing pelatihan.')
        from pytorch_forecasting.metrics import QuantileLoss
        hparams['loss'] = QuantileLoss(quantiles=list(hparams['loss'].quantiles))
        hparams['logging_metrics'] = torch.nn.ModuleList()
        model = TemporalFusionTransformer(**hparams)
        model.logging_metrics = torch.nn.ModuleList()
        model.loss = hparams['loss']
        model.inference_dataset_parameters = parameters
        model.dataset = None

        # Load state_dict (skip metrics)
        state_dict = checkpoint['state_dict']
        model_state_dict = {}

        for key, value in state_dict.items():
            if not key.startswith('logging_metrics'):
                model_state_dict[key] = value

        incompatible = model.load_state_dict(model_state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith(('loss.', 'logging_metrics.'))]
        if missing or incompatible.unexpected_keys:
            raise ValueError(f'Bobot checkpoint tidak cocok: {missing}, {incompatible.unexpected_keys}')

        # Set ke CPU dan eval mode
        model = model.cpu()
        model.eval()

        model.checkpoint_path = ckpt_path
        _LOADED_MODELS[cache_key] = model
        logging.info(f"Model {version} loaded successfully on CPU!")
        return model
    except Exception as e:
        logging.warning(f"Failed to load TFT model: {e}")
        import traceback
        logging.warning(traceback.format_exc())
        return None


def to_float(value, default=0.0):
    """Helper untuk convert ke float dengan aman (handle string, None, dll)"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


def build_input_dataframe(encoder_last_60, future_7, feature_cols=None):
    """
    Build input DataFrame dengan format PERSIS seperti training TFT.
    Dari kode: kode_tft_iqra_updated.ipynb line 607-643

    Format TimeSeriesDataSet:
    - time_idx="time_idx"
    - target="close"
    - group_ids=["group_id"]
    - min_encoder_length=30, max_encoder_length=60
    - min_prediction_length=1, max_prediction_length=7
    - static_categoricals=["group_id"]
    - time_varying_known_categoricals=["day_of_week", "month"]
    - time_varying_known_reals=["time_idx", "year_num"]
    - time_varying_unknown_reals = unknown_reals (all features)
    """
    if not isinstance(encoder_last_60, list) or len(encoder_last_60) != 60:
        raise ValueError("encoder_last_60 must be a list of 60 dicts")
    if not isinstance(future_7, list) or len(future_7) != 7:
        raise ValueError("future_7 must be a list of 7 dicts")

    # Valid days untuk bursa (no Sunday untuk beberapa data)
    VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Semua fitur unknown_reals sesuai training (NO IDR version)
    unknown_reals = UNKNOWN_REALS

    # Semua kolom yang dibutuhkan
    all_cols = unknown_reals + ["time_idx", "year_num", "day_of_week", "month", "group_id"]
    
    def add_date_features(time_idx, observation_date=None):
        """Tambah day_of_week dan month berdasarkan time_idx (asumsi mulai dari 2021-01-04)

        NOTE: Training data hanya punya 5 hari (Monday-Friday) karena stock market data.
        Weekend (Saturday/Sunday) di-map ke Friday.
        """
        import datetime
        start_date = datetime.date(2021, 1, 4)
        current_date = (pd.Timestamp(observation_date).date() if observation_date
                        else start_date + datetime.timedelta(days=time_idx - 1))

        day_name = current_date.strftime("%A")  # Monday, Tuesday, etc

        # Mapping weekend ke Friday untuk stock market data
        # Training data hanya punya weekdays (market closed on weekends)
        weekend_map = {
            "Saturday": "Friday",
            "Sunday": "Friday",
        }
        day_name = weekend_map.get(day_name, day_name)

        month_num = current_date.month
        month_num = max(1, min(12, month_num))

        return {
            "day_of_week": day_name,
            "month": str(month_num),
            "year_num": float(current_date.year)
        }
    
    # Proses encoder (60 hari)
    enc_data = []
    for i, d in enumerate(encoder_last_60):
        idx = i + 1
        date_feats = add_date_features(idx, d.get('date'))

        # Convert semua nilai ke float dengan aman
        close = to_float(d.get("close", d.get("usd_close", np.nan)))
        open_val = to_float(d.get("open", np.nan))
        tickvol = to_float(d.get("tickvol", 0))

        # Feature engineering
        close_lag1 = enc_data[-1].get("close") if enc_data else close
        close_lag5 = enc_data[-5].get("close") if len(enc_data) >= 5 else close

        # Moving averages
        closes = [x.get("close", 0) for x in enc_data]
        close_ma7 = np.mean(closes[-7:]) if len(closes) >= 7 else close
        close_ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else close
        close_std7 = np.std(closes[-7:]) if len(closes) >= 7 else 0

        # Log
        log_close = np.log(close) if close and close > 0 else 0
        log_close_lag1 = np.log(close_lag1) if close_lag1 and close_lag1 > 0 else 0
        log_close_ma7 = np.log(close_ma7) if close_ma7 and close_ma7 > 0 else 0

        # Returns
        gold_ret1 = (close - close_lag1) / close_lag1 if close_lag1 and close_lag1 > 0 else 0
        gold_ret5 = (close - close_lag5) / close_lag5 if close_lag5 and close_lag5 > 0 else 0

        # Volatility
        tickvols = [x.get("tickvol", 0) for x in enc_data]
        gold_vol5 = np.std(tickvols[-5:]) if len(tickvols) >= 5 else 0
        gold_vol20 = np.std(tickvols[-20:]) if len(tickvols) >= 20 else 0

        # USD features
        usd_close = to_float(d.get("usd_close", d.get("usd_open", np.nan)))
        usd_open = to_float(d.get("usd_open", np.nan))
        usd_adj_close = to_float(d.get("usd_adj_close", np.nan))
        usd_lag1 = enc_data[-1].get("usd_close") if enc_data else usd_close
        usd_ret1 = (usd_close - usd_lag1) / usd_lag1 if usd_lag1 and usd_lag1 > 0 else 0

        # GSPC features
        gspc_close = to_float(d.get("gspc_close", d.get("gspc_open", np.nan)))
        gspc_open = to_float(d.get("gspc_open", np.nan))
        gspc_lag1 = enc_data[-1].get("gspc_close") if enc_data else gspc_close
        gspc_ret1 = (gspc_close - gspc_lag1) / gspc_lag1 if gspc_lag1 and gspc_lag1 > 0 else 0

        # NO IDR features in no_idr version

        # Oil
        oil = to_float(d.get("oil_dcoilwtico", np.nan))
        oil_lag1 = enc_data[-1].get("oil_dcoilwtico") if enc_data else oil
        oil_ret1 = (oil - oil_lag1) / oil_lag1 if oil_lag1 and oil_lag1 > 0 else 0

        # Fed & CPI
        fed_rate = to_float(d.get("fed_rate_pct", np.nan))
        fed_target_from = to_float(d.get("fed_target_rate_from_pct", np.nan))
        fed_target_to = to_float(d.get("fed_target_rate_to_pct", np.nan))
        cpi = to_float(d.get("cpi_cpiaucsl", np.nan))
        
        # pct from MA20
        close_pct_from_ma20 = (close - close_ma20) / close_ma20 if close_ma20 and close_ma20 > 0 else 0

        row = {
            "time_idx": idx,
            "group_id": "XAU_USD",
            "year_num": date_feats["year_num"],
            "day_of_week": date_feats["day_of_week"],
            "month": date_feats["month"],
            "open": open_val if not np.isnan(open_val) else close,
            "close": close,
            "tickvol": tickvol if not np.isnan(tickvol) else 0,
            "close_lag1": close_lag1,
            "close_lag5": close_lag5,
            "close_ma7": close_ma7,
            "close_ma20": close_ma20,
            "close_std7": close_std7,
            "log_close": log_close,
            "log_close_lag1": log_close_lag1,
            "log_close_ma7": log_close_ma7,
            "close_pct_from_ma20": close_pct_from_ma20,
            "gold_ret1": gold_ret1,
            "gold_ret5": gold_ret5,
            "gold_vol5": gold_vol5,
            "gold_vol20": gold_vol20,
            "usd_open": usd_open if not np.isnan(usd_open) else usd_close,
            "usd_close": usd_close,
            "usd_adj_close": usd_adj_close if not np.isnan(usd_adj_close) else usd_close,
            "usd_close_lag1": usd_lag1,
            "usd_ret1": usd_ret1,
            "gspc_open": gspc_open if not np.isnan(gspc_open) else gspc_close,
            "gspc_close": gspc_close,
            "gspc_close_lag1": gspc_lag1,
            "gspc_ret1": gspc_ret1,
            # NO IDR features in no_idr version
            "oil_dcoilwtico": oil if not np.isnan(oil) else (enc_data[-1].get("oil_dcoilwtico", 75.0) if enc_data else 75.0),
            "oil_lag1": oil_lag1,
            "oil_ret1": oil_ret1,
            "fed_dff_pct": to_float(d.get("fed_dff_pct", d.get("fed_rate_pct", 5.33))),
            "fed_rate_pct": fed_rate if not np.isnan(fed_rate) else to_float(d.get("fed_dff_pct", 0.0525)),
            "fed_target_rate_from_pct": fed_target_from if not np.isnan(fed_target_from) else (fed_rate if not np.isnan(fed_rate) else to_float(d.get("fed_dff_pct", 0.0525))),
            "fed_target_rate_to_pct": fed_target_to if not np.isnan(fed_target_to) else (fed_rate if not np.isnan(fed_rate) else to_float(d.get("fed_dff_pct", 0.0525))),
            "cpi_cpiaucsl": cpi if not np.isnan(cpi) else 310.0,
            "usd_open_missing": 0.0,
            "usd_close_missing": 0.0,
            "gspc_open_missing": 0.0,
            "gspc_close_missing": 0.0,
            "oil_dcoilwtico_missing": 0.0,
            "fed_dff_pct_missing": 0.0,
        }
        enc_data.append(row)

    # Future data (7 hari) - gunakan last known values
    last_enc = enc_data[-1] if enc_data else {}

    for i, d in enumerate(future_7):
        idx = 61 + i
        date_feats = add_date_features(idx, d.get('date'))

        # Convert future values dengan aman
        usd_close = to_float(d.get("usd_close", last_enc.get("usd_close", np.nan)))
        usd_open = to_float(d.get("usd_open", usd_close))
        gspc_close = to_float(d.get("gspc_close", last_enc.get("gspc_close", np.nan)))
        gspc_open = to_float(d.get("gspc_open", gspc_close))
        oil = to_float(d.get("oil_dcoilwtico", last_enc.get("oil_dcoilwtico", np.nan)))
        fed_rate = to_float(d.get("fed_rate_pct", last_enc.get("fed_rate_pct", 0.09)))
        fed_target_from = to_float(d.get("fed_target_rate_from_pct", last_enc.get("fed_target_rate_from_pct", fed_rate)))
        fed_target_to = to_float(d.get("fed_target_rate_to_pct", last_enc.get("fed_target_rate_to_pct", 0.25)))
        cpi = to_float(d.get("cpi_cpiaucsl", last_enc.get("cpi_cpiaucsl", np.nan)))

        row = {
            "time_idx": idx,
            "group_id": "XAU_USD",
            "year_num": date_feats["year_num"],
            "day_of_week": date_feats["day_of_week"],
            "month": date_feats["month"],
            # Use last known values untuk future
            "open": last_enc.get("close", np.nan),
            "close": last_enc.get("close", np.nan),
            "tickvol": last_enc.get("tickvol", 0),
            "close_lag1": last_enc.get("close"),
            "close_lag5": last_enc.get("close_lag5"),
            "close_ma7": last_enc.get("close_ma7"),
            "close_ma20": last_enc.get("close_ma20"),
            "close_std7": last_enc.get("close_std7"),
            "log_close": last_enc.get("log_close"),
            "log_close_lag1": last_enc.get("log_close_lag1"),
            "log_close_ma7": last_enc.get("log_close_ma7"),
            "close_pct_from_ma20": last_enc.get("close_pct_from_ma20"),
            "gold_ret1": last_enc.get("gold_ret1"),
            "gold_ret5": last_enc.get("gold_ret5"),
            "gold_vol5": last_enc.get("gold_vol5"),
            "gold_vol20": last_enc.get("gold_vol20"),
            "usd_open": usd_open if not np.isnan(usd_open) else usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_close": usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_adj_close": usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_close_lag1": last_enc.get("usd_close"),
            "usd_ret1": last_enc.get("usd_ret1"),
            "gspc_open": gspc_open if not np.isnan(gspc_open) else gspc_close if not np.isnan(gspc_close) else last_enc.get("gspc_close", 3800.0),
            "gspc_close": gspc_close if not np.isnan(gspc_close) else last_enc.get("gspc_close", 3800.0),
            "gspc_close_lag1": last_enc.get("gspc_close"),
            "gspc_ret1": last_enc.get("gspc_ret1"),
            # NO IDR features in no_idr version
            "oil_dcoilwtico": oil if not np.isnan(oil) else last_enc.get("oil_dcoilwtico", 75.0),
            "oil_lag1": last_enc.get("oil_dcoilwtico"),
            "oil_ret1": last_enc.get("oil_ret1"),
            "fed_dff_pct": to_float(d.get("fed_dff_pct", d.get("fed_rate_pct", last_enc.get("fed_dff_pct", 5.33)))),
            "fed_rate_pct": fed_rate if not np.isnan(fed_rate) else last_enc.get("fed_rate_pct", 0.0525),
            "fed_target_rate_from_pct": fed_target_from if not np.isnan(fed_target_from) else last_enc.get("fed_target_rate_from_pct", 0.0525),
            "fed_target_rate_to_pct": fed_target_to if not np.isnan(fed_target_to) else last_enc.get("fed_target_rate_to_pct", 0.0525),
            "cpi_cpiaucsl": cpi if not np.isnan(cpi) else last_enc.get("cpi_cpiaucsl", 310.0),
            "usd_open_missing": 0.0,
            "usd_close_missing": 0.0,
            "gspc_open_missing": 0.0,
            "gspc_close_missing": 0.0,
            "oil_dcoilwtico_missing": 0.0,
            "fed_dff_pct_missing": 0.0,
        }
        enc_data.append(row)
    
    df = pd.DataFrame(enc_data)
    return df


def infer_tft_model(model, df):
    """
    Inferensi TFT model menggunakan dataset yang sudah ter-train.
    CRITICAL: Gunakan model's dataset reference untuk preservasi categorical encodings.
    """
    try:
        if hasattr(model, 'predict') and model is not None:
            logging.info("Trying TFT prediction with model's reference dataset...")

            parameters = getattr(model, 'inference_dataset_parameters', None)
            if parameters:
                from .inference_dataset import InferenceTimeSeriesDataSet
                inference_dataset = InferenceTimeSeriesDataSet.from_parameters(
                    parameters, df, predict=True, stop_randomization=True,
                )
            else:
                # Compatibility for injected dataset-backed models.
                training_dataset = getattr(model, 'dataset', None)
                if training_dataset is None:
                    logging.warning('Model has no saved preprocessing parameters')
                    return None
                if hasattr(training_dataset, 'get_parameters'):
                    from .inference_dataset import InferenceTimeSeriesDataSet
                    inference_dataset = InferenceTimeSeriesDataSet.from_dataset(
                        training_dataset, df, predict=True, stop_randomization=True,
                    )
                else:
                    inference_dataset = training_dataset.from_dataset(
                        training_dataset, df, predict=True, stop_randomization=True,
                    )

            # Buat dataloader untuk prediction
            pred_dataloader = inference_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            # Get prediction dengan mode="prediction" agar denormalized
            if parameters:
                import torch
                x, _ = next(iter(pred_dataloader))
                model.cpu().eval()
                with torch.inference_mode():
                    predictions = model.to_prediction(model(x))
            else:
                predictions = model.predict(pred_dataloader, mode="prediction", return_x=False)

            if predictions is not None and len(predictions) > 0:
                import torch
                if isinstance(predictions, torch.Tensor):
                    vals = predictions.detach().cpu().numpy().flatten()
                elif hasattr(predictions, 'numpy'):
                    vals = predictions.numpy().flatten()
                else:
                    vals = np.array(predictions).flatten()

                if len(vals) != MAX_PREDICTION_LENGTH or not np.isfinite(vals).all():
                    logging.error("TFT must return exactly seven finite predictions")
                    return None

                preds = []
                for i, v in enumerate(vals):
                    preds.append({"time_idx": int(df['time_idx'].iloc[-7 + i]), "prediction": float(v)})
                logging.info(f"TFT prediction successful: {vals[:3]}...")
                return preds
    except Exception as e:
        logging.warning(f"TFT inference error: {e}")
        import traceback
        logging.warning(traceback.format_exc())
    return None


class TFTInferenceError(RuntimeError):
    """The TFT model could not produce a complete forecast."""


def validate_manual_history(rows):
    """Require actual, dated raw observations rather than filling missing inputs."""
    import datetime
    previous = None
    for row in rows:
        date = datetime.date.fromisoformat(row.get('date', ''))
        if row['date'] != date.isoformat():
            raise ValueError('Tanggal CSV harus memakai format YYYY-MM-DD.')
        if previous is not None and date <= previous:
            raise ValueError('Tanggal CSV harus berurutan dan tidak boleh duplikat.')
        previous = date
        close_val = row.get('close')
        if close_val is None or str(close_val).strip() == '' or not np.isfinite(float(close_val)):
            raise ValueError('CSV tidak memiliki nilai aktual yang valid untuk close')
        # Ensure tickvol has a safe default
        try:
            tv = row.get('tickvol')
            if tv is None or str(tv).strip() == '' or str(tv).lower() == 'nan':
                row['tickvol'] = 0.0
            else:
                row['tickvol'] = float(tv)
        except Exception:
            row['tickvol'] = 0.0
        # Fallback DXY to USD
        if (row.get('usd_open') is None or str(row.get('usd_open')).strip() == '') and row.get('dxy_open') is not None:
            row['usd_open'] = row.get('dxy_open')
        if (row.get('usd_close') is None or str(row.get('usd_close')).strip() == '') and row.get('dxy_close') is not None:
            row['usd_close'] = row.get('dxy_close')
        if (row.get('fed_rate_pct') is None or str(row.get('fed_rate_pct')).strip() == '') and row.get('fed_dff_pct') is not None:
            row['fed_rate_pct'] = row.get('fed_dff_pct')


def infer_from_payload(payload, model=None):
    if 'historical_data' in payload:
        from .notebook_pipeline import build_notebook_forecast
        df = build_notebook_forecast(payload['historical_data'])
    else:
        df = build_input_dataframe(payload.get('encoder_last_60', []), payload.get('future_7', []))
    return infer_from_dataframe(model, df)


def infer_from_dataframe(model, df):
    if model is None:
        raise TFTInferenceError("Model TFT tidak tersedia. Tidak ada prediksi yang dihasilkan.")

    logging.info("Attempting TFT model inference...")
    try:
        preds_tft = infer_tft_model(model, df)
        if (preds_tft is not None
                and len(preds_tft) == MAX_PREDICTION_LENGTH
                and all(np.isfinite(p['prediction']) for p in preds_tft)):
            return preds_tft, "TFT-model"
    except Exception as exc:
        logging.exception("TFT inference failed")
        raise TFTInferenceError("Inferensi model TFT gagal. Tidak ada prediksi yang dihasilkan.") from exc

    raise TFTInferenceError("Model TFT tidak menghasilkan tujuh prediksi yang valid. Silakan periksa model dan data input.")


# Helper functions for analysis endpoints
def generate_sample_historical_data(days):
    """Generate sample historical data for demo"""
    import random
    data = []
    price = 2300.0
    for i in range(days):
        change = random.uniform(-20, 20)
        price = max(2000, min(2600, price + change))
        data.append({
            "time_idx": i + 1,
            "date": f"Day {i + 1}",
            "close": round(price, 2),
            "open": round(price + random.uniform(-5, 5), 2),
            "volume": random.randint(45000, 55000)
        })
    return data


def calculate_summary(data):
    """Calculate summary statistics"""
    if not data:
        return {"avg_close": 0, "min_close": 0, "max_close": 0, "volatility": 0}

    closes = [d["close"] for d in data]
    import numpy as np
    return {
        "avg_close": round(float(np.mean(closes)), 2),
        "min_close": round(float(np.min(closes)), 2),
        "max_close": round(float(np.max(closes)), 2),
        "volatility": round(float(np.std(closes)), 2)
    }


def format_data_for_frontend(data):
    """Format data for frontend consumption"""
    formatted = []
    for row in data:
        formatted.append({
            "close": float(row.get("close", 0)),
            "open": float(row.get("open", 0)),
            "tickvol": int(row.get("tickvol", 50000)),
            "usd_close": float(row.get("usd_close", 104)),
            "usd_open": float(row.get("usd_open", 104)),
            "gspc_close": float(row.get("gspc_close", 5200)),
            "gspc_open": float(row.get("gspc_open", 5200)),
            "oil_dcoilwtico": float(row.get("oil_dcoilwtico", 78)),
            "fed_rate_pct": float(row.get("fed_rate_pct", 0.0525)),
            "cpi_cpiaucsl": float(row.get("cpi_cpiaucsl", 310))
        })
    return formatted


# Cache for real-time API data (simple in-memory cache)
_API_CACHE = {
    "data": None,
    "timestamp": 0,
    "ttl": 900  # 15 minutes cache
}


def fetch_real_time_data():
    """
    Fetch real-time data from multiple free APIs
    Returns 60 days of historical data with latest values
    """
    import time
    current_time = time.time()

    # Check cache first
    if _API_CACHE["data"] is not None and (current_time - _API_CACHE["timestamp"]) < _API_CACHE["ttl"]:
        logging.info("Returning cached API data")
        return _API_CACHE["data"]

    try:
        import yfinance as yf

        logging.info("Fetching real-time data from Yahoo Finance API...")

        # Fetch gold data (GC=F is Gold Futures)
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="3mo")  # 3 months to ensure we get 60+ trading days

        # Fetch S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_hist = sp500.history(period="3mo")

        # Fetch USD Index
        usd = yf.Ticker("DX-Y.NYB")  # US Dollar Index
        usd_hist = usd.history(period="3mo")

        # Fetch Oil (WTI)
        oil = yf.Ticker("CL=F")  # Crude Oil WTI
        oil_hist = oil.history(period="3mo")

        # Take last 60 trading days
        gold_hist = gold_hist.tail(60).reset_index()
        sp500_hist = sp500_hist.tail(60).reset_index()
        usd_hist = usd_hist.tail(60).reset_index()
        oil_hist = oil_hist.tail(60).reset_index()

        # Build combined dataset
        data = []
        for i in range(len(gold_hist)):
            try:
                row = {
                    "date": str(gold_hist.iloc[i]['Date']) if 'Date' in gold_hist.columns else f"Day {i+1}",
                    "close": float(gold_hist.iloc[i]['Close']) if 'Close' in gold_hist.columns else float(gold_hist.iloc[i].get('close', 2300)),
                    "open": float(gold_hist.iloc[i]['Open']) if 'Open' in gold_hist.columns else float(gold_hist.iloc[i].get('open', 2300)),
                    "tickvol": int(gold_hist.iloc[i]['Volume']) if 'Volume' in gold_hist.columns else 50000,
                    "usd_close": float(usd_hist.iloc[i]['Close']) if i < len(usd_hist) and 'Close' in usd_hist.columns else 104.0,
                    "usd_open": float(usd_hist.iloc[i]['Open']) if i < len(usd_hist) and 'Open' in usd_hist.columns else 104.0,
                    "gspc_close": float(sp500_hist.iloc[i]['Close']) if i < len(sp500_hist) and 'Close' in sp500_hist.columns else 5200.0,
                    "gspc_open": float(sp500_hist.iloc[i]['Open']) if i < len(sp500_hist) and 'Open' in sp500_hist.columns else 5200.0,
                    "oil_dcoilwtico": float(oil_hist.iloc[i]['Close']) if i < len(oil_hist) and 'Close' in oil_hist.columns else 78.0,
                    "fed_rate_pct": 0.0525,  # Will fetch from FRED in future
                    "cpi_cpiaucsl": 310.0     # Will fetch from FRED in future
                }
                data.append(row)
            except Exception as e:
                logging.warning(f"Error processing day {i}: {e}")
                continue

        if len(data) >= 60:
            # Cache the result
            _API_CACHE["data"] = data
            _API_CACHE["timestamp"] = current_time

            logging.info(f"Successfully fetched {len(data)} days from Yahoo Finance API")
            return data
        else:
            logging.warning(f"Only got {len(data)} days from API, need 60")
            return None

    except ImportError as e:
        logging.warning("yfinance not installed, unable to fetch live market data")
        raise RuntimeError("yfinance is not installed. Install yfinance>=0.2.28 and restart the app.") from e
    except Exception as e:
        logging.error(f"Error fetching real-time data: {e}")
        raise RuntimeError(f"Failed to fetch real-time data from Yahoo Finance: {e}") from e


def build_api_market_payload():
    """Build a ready-to-use payload from real market data via yfinance."""
    api_data = fetch_real_time_data()
    if not api_data or len(api_data) < 60:
        raise RuntimeError("Insufficient real-time market data returned from Yahoo Finance (need 60 rows).")

    encoder_last_60 = api_data[-60:]
    last_row = encoder_last_60[-1]
    future_7 = []

    for _ in range(7):
        future_7.append({
            "usd_close": float(last_row.get("usd_close", 104.0)),
            "usd_open": float(last_row.get("usd_open", last_row.get("usd_close", 104.0))),
            "gspc_close": float(last_row.get("gspc_close", 5200.0)),
            "gspc_open": float(last_row.get("gspc_open", last_row.get("gspc_close", 5200.0))),
            "oil_dcoilwtico": float(last_row.get("oil_dcoilwtico", 78.0)),
            "fed_rate_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "fed_target_rate_from_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "fed_target_rate_to_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "cpi_cpiaucsl": float(last_row.get("cpi_cpiaucsl", 310.0))
        })

    return {
        "encoder_last_60": encoder_last_60,
        "future_7": future_7,
        "source": "Yahoo Finance API",
        "data_points": len(encoder_last_60)
    }


def create_app():
    from flask import Flask, jsonify, request, render_template
    import logging

    logging.basicConfig(level=logging.INFO)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_path, "templates")
    static_path = os.path.join(base_path, "static")
    
    app = Flask(__name__, template_folder=template_path, static_folder=static_path)

    def _full_inference_available():
        try:
            import pytorch_forecasting
            return True
        except:
            return False

    @app.route("/health", methods=["GET"])
    def health():
        versions = discover_versions()
        loaded = [v for v in versions if os.path.abspath(best_ckpt_for_version(v) or '') in _LOADED_MODELS]
        return jsonify({
            "status": "ok", 
            "versions": versions, 
            "loaded": loaded,
            "tft_available": _full_inference_available()
        })

    @app.route("/status", methods=["GET"])
    def status():
        versions = discover_versions()
        status_map = {}
        models_loaded = []

        for v in versions:
            m = load_model(v)
            is_loaded = m is not None
            status_map[v] = {"loaded": is_loaded}
            if is_loaded:
                models_loaded.append(v)

        return jsonify({
            "versions": status_map,
            "models": models_loaded,  # For frontend compatibility
            "full_inference_available": _full_inference_available()
        })

    @app.route("/versions", methods=["GET"])
    def versions():
        return jsonify({"versions": discover_versions()})

    @app.route('/load-dataset', methods=['GET'])
    def load_dataset():
        from .scenarios import read_history_csv
        from .notebook_pipeline import build_notebook_forecast, RAW_COLUMNS
        try:
            frame, source, _ = read_history_csv()
            forecast = build_notebook_forecast(frame)
            rows = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
            return jsonify(success=True, data=rows, source=source,
                           date_range=[str(frame['date'].min()), str(frame['date'].max())],
                           last_observation_date=forecast.iloc[-8]['date'].strftime('%Y-%m-%d'))
        except (ValueError, OSError, KeyError) as exc:
            return jsonify(success=False, error=str(exc)), 400

    @app.route("/api/market-data", methods=["GET"])
    def api_market_data():
        """Fetch live market data using Yahoo Finance and build TFT payload."""
        try:
            payload = build_api_market_payload()
            return jsonify({"success": True, "payload": payload})
        except Exception as e:
            logging.error(f"Market data endpoint failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True) or {}
        if 'historical_data' in data:
            from .notebook_pipeline import build_notebook_forecast
            version = data.get('version', 'version_0')
            try:
                if not isinstance(data['historical_data'], list):
                    raise ValueError('historical_data harus berupa daftar observasi CSV.')
                adjustment = data.get('forecast_adjustment', 'none')
                if adjustment != 'none':
                    raise ValueError('forecast_adjustment model lama tidak berlaku pada model training versi 2; gunakan none.')
                df = build_notebook_forecast(data['historical_data'])
                preds, mode = infer_from_dataframe(load_model(version), df)
                adjustment_info = None
            except TFTInferenceError as exc:
                return jsonify(error=str(exc)), 503
            except (ValueError, TypeError, KeyError) as exc:
                return jsonify(error='Data CSV tidak valid: ' + str(exc)), 400
            future_rows = df.tail(7)
            predictions = [dict(pred, date=row.date.strftime('%Y-%m-%d'))
                           for pred, row in zip(preds, future_rows.itertuples())]
            from .forecast_actuals import attach_actuals
            predictions = attach_actuals(predictions)
            return jsonify(version=version, mode=mode, predictions=predictions,
                           adjustment=adjustment_info,
                           preprocessing='ID8_run3_LR0,0005,BS_32vers2.ipynb',
                           last_observation_date=df.iloc[-8]['date'].strftime('%Y-%m-%d'))
        encoder60 = data.get("encoder_last_60", [])
        future7 = data.get("future_7", [])

        if not isinstance(encoder60, list) or len(encoder60) != 60:
            return jsonify({"error": "encoder_last_60 must be 60 items"}), 400
        if not isinstance(future7, list) or len(future7) != 7:
            return jsonify({"error": "future_7 must be 7 items"}), 400

        if data.get('input_mode') in ('manual', 'upload', 'sample'):
            try:
                validate_manual_history(encoder60)
                previous = pd.Timestamp(encoder60[-1]['date'])
                for row in future7:
                    date = pd.Timestamp(row['date'])
                    if pd.isna(date) or date <= previous:
                        raise ValueError('Tanggal horizon harus sesudah observasi terakhir.')
                    previous = date
                    for key in ['usd_open', 'usd_close', 'gspc_open', 'gspc_close',
                                'oil_dcoilwtico', 'fed_rate_pct', 'cpi_cpiaucsl',
                                'fed_target_rate_from_pct', 'fed_target_rate_to_pct']:
                        value = row.get(key)
                        if value is None or str(value).strip() == '' or not np.isfinite(float(value)):
                            raise ValueError('Asumsi horizon tidak valid untuk ' + key)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                return jsonify(error='Data input manual tidak valid: ' + str(exc)), 400

        version = data.get("version", "version_0")
        # Run inference
        payload = {
            "version": version,
            "encoder_last_60": encoder60,
            "future_7": future7
        }

        try:
            model = load_model(version)
            preds, mode = infer_from_payload(payload, model)
        except TFTInferenceError as exc:
            return jsonify({"version": version, "error": str(exc)}), 503
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": "Data input tidak valid: " + str(exc)}), 400

        logging.info(f"Prediction mode: {mode}")

        if all(row.get('date') for row in future7):
            preds = [dict(prediction, date=future7[i]['date']) for i, prediction in enumerate(preds)]

        from .forecast_actuals import attach_actuals
        preds = attach_actuals(preds)

        return jsonify({
            "version": version,
            "predictions": preds,
            "mode": mode
        })

    @app.route("/upload-csv", methods=["POST"])
    def upload_csv():
        """Upload dan validasi CSV file untuk historical data"""
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        try:
            # Baca CSV
            import io
            content = file.read().decode('utf-8')
            lines = content.split('\n')

            if len(lines) < 2:
                return jsonify({"error": "CSV is empty or has no data rows"}), 400

            # Parse headers
            headers = [h.strip().lower() for h in lines[0].split(',')]

            # Validasi required columns
            required_cols = ['close']
            missing_cols = [col for col in required_cols if col not in headers]

            if missing_cols:
                return jsonify({
                    "error": f"Missing required columns: {', '.join(missing_cols)}",
                    "required": required_cols,
                    "found": headers
                }), 400

            # Parse data rows
            rows = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                values = line.split(',')
                row = {}
                for i, h in enumerate(headers):
                    if i < len(values):
                        val = values[i].strip()
                        try:
                            row[h] = float(val)
                        except:
                            row[h] = val

                if 'close' in row and row['close']:
                    rows.append(row)

            if len(rows) < 60:
                return jsonify({
                    "error": f"CSV must have at least 60 data rows. Found: {len(rows)}",
                    "rows_found": len(rows),
                    "rows_required": 60
                }), 400

            # Ambil last 60 rows
            last_60 = rows[-60:]

            return jsonify({
                "success": True,
                "total_rows": len(rows),
                "encoder_last_60": last_60,
                "preview": rows[:10],  # First 10 rows for preview
                "headers": headers
            })

        except Exception as e:
            logging.error(f"CSV upload error: {e}")
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 500

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        """Display fixed reference evaluation values, independently of live weights."""
        from .notebook_evaluation import read_evaluation
        try:
            result, _ = read_evaluation()
            return jsonify(result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return jsonify(success=False, error=f'Hasil evaluasi tidak tersedia: {exc}'), 503

    @app.route('/evaluation-notebook-chart', methods=['GET'])
    def evaluation_notebook_chart():
        from flask import Response
        from .notebook_evaluation import read_evaluation_chart
        try:
            chart = read_evaluation_chart()
            return Response(chart, mimetype='image/png')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return jsonify(success=False, error=f'Grafik evaluasi tidak tersedia: {exc}'), 503

    @app.route("/historical-performance", methods=["GET"])
    def historical_performance():
        """
        Get historical performance data for visualization
        Useful for thesis Chapter 4 - Results
        """
        try:
            version = request.args.get("version", "version_0")
            days = min(int(request.args.get("days", 30)), 100)  # Cap at 100 days

            # Load real historical data from local CSV
            from pathlib import Path
            import pandas as pd
            root = Path(__file__).resolve().parents[1]
            csv_path = root / 'gold_price_no_idr.csv'
            if not csv_path.is_file():
                csv_path = root / 'gold_price_no_idr_update.csv'

            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                df = df.tail(days).reset_index(drop=True)
                historical_data = []
                for idx in range(len(df)):
                    row_data = df.iloc[idx]
                    date_val = str(row_data.get('date', f'Day {idx+1}'))
                    close_val = float(row_data.get('close', 0))
                    open_val = float(row_data.get('open', close_val))
                    historical_data.append({
                        "time_idx": int(idx),
                        "date": date_val,
                        "close": close_val,
                        "open": open_val,
                        "volume": 50000
                    })
                return jsonify({
                    "success": True,
                    "data": historical_data,
                    "summary": calculate_summary(historical_data)
                })

            # Fallback if CSV not found
            historical_data = generate_sample_historical_data(days)
            return jsonify({
                "success": True,
                "data": historical_data,
                "summary": calculate_summary(historical_data),
                "note": "Sample data"
            })

        except Exception as e:
            logging.error(f"Historical performance error: {e}")
            import traceback
            # Return sample data on error
            historical_data = generate_sample_historical_data(days)
            return jsonify({
                "success": True,
                "data": historical_data,
                "summary": calculate_summary(historical_data),
                "note": f"Sample data - error: {str(e)[:100]}"
            })

    @app.route("/model-info", methods=["GET"])
    def model_info():
        """
        Get model information for thesis documentation
        Architecture, hyperparameters, training details
        """
        try:
            version = request.args.get("version", "version_0")

            # Load model to get info
            model = load_model(version)
            if model is None:
                return jsonify({"error": "Model not available"}), 404

            # Get checkpoint info
            ckpt_path = best_ckpt_for_version(version)
            ckpt_dir = os.path.dirname(ckpt_path)
            pkl_path = os.path.join(ckpt_dir, "training_dataset.pkl")

            # Model architecture details
            model_info = {
                "model_type": "Temporal Fusion Transformer (TFT)",
                "version": version,
                "checkpoint": os.path.basename(ckpt_path),
                "training": {"learning_rate": float(model.hparams.learning_rate)},
                "architecture": {
                    "encoder_length": MAX_ENCODER_LENGTH,
                    "decoder_length": MAX_PREDICTION_LENGTH,
                    "hidden_size": getattr(model.hparams, "hidden_size", 16),
                    "attention_head_size": getattr(model.hparams, "attention_head_size", 4),
                    "dropout": getattr(model.hparams, "dropout", 0.1),
                    "hidden_continuous_size": getattr(model.hparams, "hidden_continuous_size", 8),
                    "loss": str(getattr(model.hparams, "loss", "QuantileLoss"))
                },
                "features": {
                    "target": TARGET,
                    "group_ids": GROUP_IDS,
                    "static_categoricals": STATIC_CATEGORICALS,
                    "time_varying_known_categoricals": TIME_VARYING_KNOWN_CATEGORICALS,
                    "time_varying_known_reals": TIME_VARYING_KNOWN_REALS,
                    "time_varying_unknown_reals": getattr(model, 'inference_dataset_parameters', {}).get('time_varying_unknown_reals', UNKNOWN_REALS)
                }
            }

            import json
            from pathlib import Path
            manifest = Path(ckpt_path).parent.parent / 'source.json'
            if manifest.is_file():
                model_info['source'] = json.loads(manifest.read_text())
                model_info['training']['batch_size'] = model_info['source'].get('batch_size')

            return jsonify({
                "success": True,
                "model": model_info
            })

        except Exception as e:
            logging.error(f"Model info error: {e}")
            return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

    @app.route('/sample-scenarios', methods=['GET'])
    def sample_scenarios():
        from .scenarios import load_scenarios
        try:
            return jsonify(load_scenarios())
        except (ValueError, OSError, KeyError) as exc:
            return jsonify(success=False, error=str(exc)), 400


    @app.route("/api/refresh-data", methods=["POST"])
    def refresh_real_time_data():
        """
        Force refresh real-time data from API
        Clears cache and fetches fresh data
        """
        try:
            import time

            # Clear cache
            _API_CACHE["data"] = None
            _API_CACHE["timestamp"] = 0

            # Fetch fresh data
            data = fetch_real_time_data()

            if data and len(data) >= 60:
                # Get latest values
                latest = data[-1]
                return jsonify({
                    "success": True,
                    "message": "Real-time data refreshed successfully",
                    "source": "Yahoo Finance API",
                    "latest": {
                        "gold_price": round(latest["close"], 2),
                        "usd_index": round(latest["usd_close"], 2),
                        "sp500": round(latest["gspc_close"], 2),
                        "oil_price": round(latest["oil_dcoilwtico"], 2),
                        "data_points": len(data),
                        "last_updated": latest.get("date", "N/A")
                    }
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to fetch real-time data. Using cached/local data instead."
                }), 400

        except Exception as e:
            logging.error(f"Refresh error: {e}")
            return jsonify({"error": f"Failed to refresh: {str(e)}"}), 500


    @app.route("/", methods=["GET"])
    def index():
        versions = discover_versions()
        return render_template("index.html", versions=versions)

    return app
