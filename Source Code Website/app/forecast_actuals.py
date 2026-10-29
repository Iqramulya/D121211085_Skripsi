"""Match forecasts to observed closing prices without filling missing dates."""
from pathlib import Path

import numpy as np
import pandas as pd


def attach_actuals(predictions, csv_path=None):
    path = Path(csv_path) if csv_path else Path(__file__).resolve().parents[1] / 'gold_price_no_idr.csv'
    actuals = {}
    if csv_path is None:
        # Observed comparison prices saved in cell 29 of the supplied notebook.
        reference = Path(__file__).resolve().parents[1] / 'models/training_v2/forecast_actuals.csv'
        try:
            for row in pd.read_csv(reference).itertuples(index=False):
                actuals[str(row.date)] = float(row.close)
        except (OSError, ValueError):
            pass
    try:
        history = pd.read_csv(path, usecols=['date', 'close'])
        for row in history.itertuples(index=False):
            date = pd.to_datetime(row.date, errors='coerce')
            price = pd.to_numeric(row.close, errors='coerce')
            if pd.notna(date) and np.isfinite(price) and price > 0:
                actuals[date.strftime('%Y-%m-%d')] = float(price)
    except (OSError, ValueError):
        pass  # Missing observations must not prevent forecasting.

    result = []
    for prediction in predictions:
        date = pd.to_datetime(prediction.get('date'), errors='coerce')
        actual = actuals.get(date.strftime('%Y-%m-%d')) if pd.notna(date) else None
        absolute_error = abs(float(prediction['prediction']) - actual) if actual is not None else None
        percentage_error = (absolute_error / actual * 100
                            if actual is not None and np.isfinite(actual) and actual > 0 else None)
        result.append(dict(prediction, actual_price=actual,
                           absolute_error=absolute_error,
                           percentage_error=percentage_error))
    return result
