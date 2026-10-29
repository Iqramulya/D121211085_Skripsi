"""Saved validation outputs from the fifth training run, version 2."""
import csv
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'models/training_v2/hasil_evaluasi_tft'
CHART_PATH = RESULTS / 'grafik_evaluasi_multi_horizon_tft.png'
METRICS_PATH = RESULTS / 'hasil_evaluasi_multi_horizon_tft.csv'
HORIZON_COUNT = 7
SEQUENCE_COUNT = 894


def _validate_horizons(horizons):
    """Reject incomplete or corrupt artifacts before exposing saved metrics."""
    if len(horizons) != HORIZON_COUNT:
        raise ValueError('Evaluasi harus memuat tujuh horizon H+1 hingga H+7.')
    for day, row in enumerate(horizons, 1):
        if row['horizon'] != f'H+{day}' or row['day'] != day:
            raise ValueError('Urutan atau identitas horizon evaluasi tidak valid.')
        if row['sample_size'] != SEQUENCE_COUNT:
            raise ValueError('Jumlah sampel evaluasi berbeda dari run training versi 2.')
        if not all(math.isfinite(row[key]) for key in ('mae', 'rmse', 'mape', 'r2_score')):
            raise ValueError('Metrik evaluasi harus berupa angka finite.')
        if any(row[key] < 0 for key in ('mae', 'rmse', 'mape')):
            raise ValueError('Metrik error evaluasi tidak boleh negatif.')


def read_evaluation():
    raw = METRICS_PATH.read_bytes()
    rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    horizons = [dict(horizon=row['Horizon'], day=int(row['Hari_ke']),
                     sample_size=int(row['Jumlah_Sampel']),
                     mae=float(row['MAE']), rmse=float(row['RMSE']),
                     mape=float(row['MAPE_Persen']), r2_score=float(row['R2']))
                for row in rows]
    _validate_horizons(horizons)
    return {
        'success': True,
        'source': 'Train ke-5 / versi 2 — ID8 run3 (hasil validasi notebook)',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'evaluation_mode': 'saved_training_validation',
        # Overall values printed in cell 21, not an average of per-horizon R².
        'metrics': {'mae': 68.5037, 'rmse': 107.6815, 'mape': 1.97, 'r2_score': 0.9884},
        'sample_size': sum(row['sample_size'] for row in horizons),
        'sequence_count': SEQUENCE_COUNT, 'horizon_count': HORIZON_COUNT,
        'validation_range': ['2023-06-08', '2026-10-12'],
        'multi_horizon': horizons, 'chart_url': '/evaluation-notebook-chart',
    }, None


def read_evaluation_chart():
    return CHART_PATH.read_bytes()
