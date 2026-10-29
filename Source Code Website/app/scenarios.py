"""Deterministic selection of observed 60-row market regimes from a local CSV."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .notebook_pipeline import RAW_COLUMNS as NOTEBOOK_COLUMNS
RAW_COLUMNS = NOTEBOOK_COLUMNS[1:]


def _json_records(frame):
    """Represent missing/non-finite CSV values as JSON null, including extra columns."""
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return clean.astype(object).where(clean.notna(), None).to_dict('records')


def build_scenarios(frame, source):
    required = ['date'] + RAW_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Kolom historis belum lengkap: ' + ', '.join(sorted(missing)))
    raw = frame.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='raise')
    raw = raw.sort_values('date').reset_index(drop=True)
    if raw['date'].isna().any() or raw['date'].duplicated().any():
        raise ValueError('Tanggal historis tidak valid atau duplikat.')
    raw[RAW_COLUMNS] = raw[RAW_COLUMNS].apply(pd.to_numeric, errors='coerce')
    from .notebook_pipeline import prepare_history
    frame = prepare_history(raw)
    if len(frame) < 60 or (frame['close'] <= 0).any():
        raise ValueError('Dibutuhkan minimal 80 observasi mentah untuk fitur notebook dan encoder 60 observasi.')
    raw['date'] = raw['date'].dt.strftime('%Y-%m-%d')
    frame['date'] = frame['date'].dt.strftime('%Y-%m-%d')
    candidates = []
    for start in range(len(frame) - 59):
        prices = frame['close'].iloc[start:start + 60].to_numpy()
        stats = {
            'change_pct': float((prices[-1] / prices[0] - 1) * 100),
            'range_pct': float((prices.max() - prices.min()) / prices[0] * 100),
            'daily_volatility_pct': float(np.std(prices[1:] / prices[:-1] - 1) * 100),
        }
        candidates.append((start, stats))

    rules = {
        'bullish': (lambda s: s['change_pct'] >= 3, lambda s: -s['change_pct'], 'Bullish'),
        'bearish': (lambda s: s['change_pct'] <= -3, lambda s: s['change_pct'], 'Bearish'),
        'sideways': (lambda s: abs(s['change_pct']) <= 2 and s['range_pct'] <= 8,
                     lambda s: (abs(s['change_pct']), s['range_pct']), 'Sideways'),
        'volatile': (lambda s: s['daily_volatility_pct'] >= 1,
                     lambda s: -s['daily_volatility_pct'], 'Volatile'),
    }
    scenarios, unavailable = {}, {}
    for key, (matches, score, name) in rules.items():
        eligible = [(start, stats) for start, stats in candidates if matches(stats)]
        if not eligible:
            unavailable[key] = f'Tidak ada periode 60 observasi yang memenuhi kriteria {name} pada CSV ini.'
            continue
        start, stats = min(eligible, key=lambda item: (score(item[1]), item[0]))
        rows = _json_records(frame.iloc[start:start + 60])
        scenarios[key] = {
            'name': name, 'data': rows, 'stats': stats,
            'history': _json_records(raw.loc[raw['date'] <= rows[-1]['date']]),
            'description': (f"{rows[0]['date']} s.d. {rows[-1]['date']}; perubahan "
                            f"{stats['change_pct']:+.2f}%; volatilitas return harian "
                            f"{stats['daily_volatility_pct']:.2f}%"),
        }
    return {'success': True, 'source': source, 'scenarios': scenarios, 'unavailable': unavailable,
            'rules': {'bullish': 'Return 60 observasi >= 3%', 'bearish': 'Return <= -3%',
                      'sideways': 'Absolute return <= 2% dan rentang harga <= 8%',
                      'volatile': 'Standar deviasi return harian >= 1%'},
            'note': 'Periode dipilih dari CSV dan preprocessing notebook.'}


def read_history_csv():
    explicit = os.environ.get('TFT_SCENARIO_CSV')
    if explicit:
        path = Path(explicit)
    else:
        root = Path(__file__).resolve().parents[1]
        reference_csv = root / 'models' / 'training_v2' / 'gold_price_no_idr_update.csv'
        direct_csv = reference_csv if reference_csv.is_file() else root / 'gold_price_no_idr.csv'
        if direct_csv.is_file():
            path = direct_csv
        else:
            candidates = sorted({p.resolve() for directory in [root, root.parent, root.parent.parent]
                                 for p in directory.glob('gold_price_no_idr*.csv')})
            if not candidates:
                raise ValueError('CSV sumber skenario tidak ditemukan. Periksa TFT_SCENARIO_CSV.')
            path = candidates[0]
    if not path.is_file():
        raise ValueError('CSV sumber skenario tidak ditemukan. Periksa TFT_SCENARIO_CSV.')
    raw = path.read_bytes()
    from io import BytesIO
    frame = pd.read_csv(BytesIO(raw))
    source = path.name
    return frame, source, hashlib.sha256(raw).hexdigest()


def load_scenarios():
    frame, source, digest = read_history_csv()
    result = build_scenarios(frame, source)
    result['source_id'] = digest
    return result
