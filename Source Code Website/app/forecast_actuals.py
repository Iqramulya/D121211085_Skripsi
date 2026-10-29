"""Attach exact-date observations without changing or rerunning a forecast."""
from datetime import date, datetime, time, timedelta
import logging
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


_SPECIAL_START = date(2026, 10, 13)
_SPECIAL_END = date(2026, 10, 21)
_SPECIAL_CSV = Path(__file__).resolve().parents[1] / 'aktual.csv'
_NEW_YORK = ZoneInfo('America/New_York')


def _now_new_york():
    """Keep the completed-daily-bar cutoff independently patchable."""
    return datetime.now(_NEW_YORK)


def _parse_date(value):
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _is_special(day):
    return _SPECIAL_START <= day <= _SPECIAL_END


def _positive_price(value):
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _csv_actuals(path):
    """Read all valid observations once for coverage and exact-date attachment."""
    actuals = {}
    try:
        history = pd.read_csv(path, usecols=['date', 'close'])
        for row in history.itertuples(index=False):
            day = _parse_date(row.date)
            price = _positive_price(row.close)
            if day is not None and price is not None:
                actuals[day] = price
    except (OSError, ValueError, TypeError) as exc:
        logging.warning('Special actual prices unavailable from %s (%s)',
                        Path(path).name, type(exc).__name__)
    return actuals


def actuals_coverage():
    """Describe local actual availability without contacting Yahoo or a model."""
    actuals = _csv_actuals(_SPECIAL_CSV)
    return {
        'special_start': _SPECIAL_START.isoformat(),
        'special_end': _SPECIAL_END.isoformat(),
        'csv': {
            'available': bool(actuals),
            'first_date': min(actuals).isoformat() if actuals else None,
            'last_date': max(actuals).isoformat() if actuals else None,
            'special_dates': sorted(day.isoformat() for day in actuals if _is_special(day)),
        },
    }


def _yahoo_actuals(days):
    if not days:
        return {}
    actuals = {}
    try:
        import yfinance as yf

        ticker = yf.Ticker('GC=F')
        options = dict(start=min(days).isoformat(),
                       end=(max(days) + timedelta(days=1)).isoformat(),
                       interval='1d', auto_adjust=False, timeout=10)
        history = ticker.history(**options)
        for index, value in history['Close'].items():
            stamp = pd.Timestamp(index)
            day = stamp.date() if pd.notna(stamp) else None
            price = _positive_price(value)
            if day in days and price is not None:
                actuals[day] = price
    except Exception as exc:
        # Provider/network failures must never discard the existing TFT forecast.
        logging.warning('Yahoo Finance actual prices unavailable: %s', exc)
    return actuals


def attach_actuals(predictions, csv_path=None):
    """Use the special CSV only for October 13–21, otherwise GC=F daily Close."""
    dates = [_parse_date(prediction.get('date')) for prediction in predictions]
    special_days = {day for day in dates if day is not None and _is_special(day)}
    csv_actuals = _csv_actuals(Path(csv_path) if csv_path is not None else _SPECIAL_CSV) if special_days else {}
    actuals = {day: price for day, price in csv_actuals.items() if day in special_days}
    now = _now_new_york()
    eligible_days = {
        day for day in dates
        if day is not None and not _is_special(day)
        and (day < now.date() or (day == now.date() and now.time() >= time(17)))
    }
    actuals.update(_yahoo_actuals(eligible_days))

    result = []
    for prediction, day in zip(predictions, dates):
        actual = actuals.get(day)
        absolute_error = abs(float(prediction['prediction']) - actual) if actual is not None else None
        percentage_error = absolute_error / actual * 100 if actual is not None else None
        result.append(dict(prediction, actual_price=actual,
                           absolute_error=absolute_error,
                           percentage_error=percentage_error))
    return result
