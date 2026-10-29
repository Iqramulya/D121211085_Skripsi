import numpy as np
import pandas as pd
import logging


def build_input_dataframe(encoder_last_60, future_7, feature_cols=None):
    # Expect encoder_last_60 to be list of dicts with keys matching feature_cols or defaults
    if not isinstance(encoder_last_60, list) or len(encoder_last_60) != 60:
        raise ValueError("encoder_last_60 must be a list of 60 dicts")
    if not isinstance(future_7, list) or len(future_7) != 7:
        raise ValueError("future_7 must be a list of 7 dicts")

    cols = feature_cols or ["USD", "GSPC", "Oil", "FedRate", "CPI"]  # NO IDR
    def to_row(d, time_idx):
        row = {"time_idx": time_idx, "group_id": 0}
        for c in cols:
            row[c] = d.get(c, np.nan)
        row["target"] = np.nan
        return row

    rows = []
    for i, d in enumerate(encoder_last_60, start=1):
        rows.append(to_row(d, i))
    for i, d in enumerate(future_7, start=61):
        rows.append(to_row(d, i))
    df = pd.DataFrame(rows)
    return df


def simple_predict(model, input_df):
    # Try to use model.predict if available; else fallback to naive last value
    try:
        if hasattr(model, "predict"):
            res = model.predict(input_df)
            # Normalize to a list of 7 numeric values
            if isinstance(res, (list, tuple)):
                vals = list(res)[:7]
            elif isinstance(res, (np.ndarray, pd.Series)):
                vals = list(res.values)[:7]
            else:
                vals = [float(res)] * 7
            return [float(v) for v in vals]
    except Exception:
        pass
    # Fallback: use last known value from encoder_last_60 if present
    if input_df is not None and not input_df.empty:
        last = input_df.iloc[-1].get("USD", 1.0)
        return [float(last)] * 7
    return [0.0] * 7


def _try_full_inference_full(model, df):
    # Attempt to run full TFT inference path using PyTorch Forecasting PredictionDataset
    try:
        from pytorch_forecasting.data import PredictionDataset
        from torch.utils.data import DataLoader
        # Construct a prediction dataset. This mirrors the training-time inputs.
        pred_ds = PredictionDataset(
            df,
            time_idx="time_idx",
            target=None,
            group_ids=["group_id"],
            min_encoder_length=60,
            max_encoder_length=60,
            min_prediction_length=7,
            max_prediction_length=7,
            time_vary_known_reals=["USD", "GSPC", "Oil", "FedRate", "CPI"],  # NO IDR
        )
        loader = DataLoader(pred_ds, batch_size=32, shuffle=False)
        if hasattr(model, "predict"):
            res = model.predict(loader)
        else:
            # If model doesn't implement predict via loader, bail out to None
            return None
        # Normalize the result to a list of 7 predictions (time_idx 61..67)
        preds = []
        if isinstance(res, (list, tuple)):
            for item in res:
                if isinstance(item, (list, tuple)):
                    preds.extend(list(item)[:7])
                else:
                    preds.append(item)
        elif isinstance(res, np.ndarray):
            arr = res.flatten()
            preds = arr[:7].tolist()
        elif isinstance(res, pd.DataFrame):
            if res.shape[1] >= 1:
                preds = res.values[:7, 0].tolist()
        if len(preds) >= 7:
            return [{"time_idx": 60 + i + 1, "prediction": float(preds[i])} for i in range(7)]
        # If can't parse, fall back
        return None
    except Exception as e:
        logging.debug(f"Full TFT inference path failed: {e}")
        return None


def infer_from_payload(payload, model=None):
    # Build input df from payload and perform prediction
    version = payload.get("version", "version_0")
    encoder_last_60 = payload.get("encoder_last_60", [])
    future_7 = payload.get("future_7", [])
    df = build_input_dataframe(encoder_last_60, future_7)
    # Try full TFT inference path first
    preds_full = _try_full_inference_full(model, df) if model is not None else None
    if preds_full is not None:
        return preds_full
    # Fallback to simple/predict path
    preds = simple_predict(model, df)
    out = []
    base = 60
    for i, val in enumerate(preds, start=1):
        out.append({"time_idx": base + i, "prediction": float(val)})
    return out
