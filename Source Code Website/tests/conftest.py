"""Keep forecast comparisons independent of external data and wall-clock time."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app import forecast_actuals


@pytest.fixture(autouse=True)
def isolated_forecast_actual_sources(monkeypatch, tmp_path, request):
    monkeypatch.setattr(forecast_actuals, '_SPECIAL_CSV', tmp_path / 'missing-aktual.csv')
    monkeypatch.setattr(forecast_actuals, '_now_new_york', lambda: datetime(
        2026, 10, 29, 16, tzinfo=ZoneInfo('America/New_York')))

    # Adapter regressions supply their own fake Ticker; all other forecast tests
    # isolate this comparison boundary without changing market-data behavior.
    if request.module.__name__.rsplit('.', 1)[-1] != 'test_forecast_actuals':
        monkeypatch.setattr(forecast_actuals, '_yahoo_actuals', lambda days: {})
