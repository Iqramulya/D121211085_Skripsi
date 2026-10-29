"""Explicit experimental alignment to one saved Colab forecast origin."""
OFFSET_USD = -216.71723144531234


def adjust_forecast(predictions):
    adjusted = [dict(row, raw_prediction=row['prediction'],
                     prediction=row['prediction'] + OFFSET_USD,
                     adjustment_usd=OFFSET_USD) for row in predictions]
    if any(row['prediction'] <= 0 for row in adjusted):
        raise ValueError('Koreksi percobaan menghasilkan harga nonpositif; gunakan hasil model asli.')
    return adjusted, {
        'method': 'experimental_reference_offset', 'offset_usd': OFFSET_USD,
        'reference_notebook': 'TFT_Optuna_Multi_Horizon_Validation_20_Persenp.ipynb',
        'fit_origin': '2026-02-18', 'fit_horizons': [1, 2, 3, 4],
        'note': 'Koreksi terhadap output notebook; akurasi harga aktual dan tanggal lain belum divalidasi.',
    }
