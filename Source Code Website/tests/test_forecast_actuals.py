from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yfinance as yf

import app
from app import forecast_actuals
from app.forecast_actuals import attach_actuals


@pytest.fixture
def yahoo(monkeypatch):
    state = {'frame': pd.DataFrame(columns=['Close']), 'calls': [], 'symbols': []}

    class Ticker:
        def __init__(self, symbol):
            state['symbols'].append(symbol)

        def history(self, **kwargs):
            state['calls'].append(kwargs)
            if state.get('error'):
                raise state['error']
            return state['frame']

    monkeypatch.setattr(yf, 'Ticker', Ticker)
    return state


def closes(dates, prices):
    return pd.DataFrame({'Close': prices}, index=pd.DatetimeIndex(dates, tz='America/New_York'))


def assert_missing(row):
    assert row['actual_price'] is None
    assert row['absolute_error'] is None
    assert row['percentage_error'] is None


def test_actuals_match_dates_and_displayed_forecast(tmp_path, yahoo):
    source = tmp_path / 'prices.csv'
    source.write_text('date,close\n2026-10-13,2000\n2026-10-14,NaN\n2026-10-15T12:00:00,2100\n')
    predictions = [
        {'date': '2026-10-13', 'prediction': 1980, 'raw_prediction': 2200},
        {'date': '2026-10-14', 'prediction': 1990},
        {'date': '2026-10-15', 'prediction': 2010},
        {'prediction': 2020},
    ]
    result = attach_actuals(predictions, source)
    assert result[0]['actual_price'] == 2000
    assert result[0]['absolute_error'] == 20
    assert result[0]['percentage_error'] == 1
    assert result[0]['raw_prediction'] == 2200
    for row in result[1:]:
        assert_missing(row)
    assert 'actual_price' not in predictions[0]
    assert not yahoo['calls']


def test_special_boundaries_and_source_priority(tmp_path, yahoo):
    source = tmp_path / 'aktual.csv'
    dates = ['2026-10-12', '2026-10-13', '2026-10-21', '2026-10-22']
    source.write_text('date,close\n' + ''.join(f'{day},2000\n' for day in dates))
    yahoo['frame'] = closes(dates, [3000, 3100, 3200, 3300])
    result = attach_actuals([{'date': day, 'prediction': 1980} for day in dates], source)
    assert [row['actual_price'] for row in result] == [3000, 2000, 2000, 3300]
    assert yahoo['symbols'] == ['GC=F']
    assert len(yahoo['calls']) == 1
    options = yahoo['calls'][0]
    assert options['start'] == '2026-10-12'
    assert options['end'] == '2026-10-23'
    assert options['interval'] == '1d'
    assert options['auto_adjust'] is False
    assert 0 < options['timeout'] <= 15


def test_default_special_source_is_root_aktual_only(monkeypatch, tmp_path, yahoo):
    source = tmp_path / 'aktual.csv'
    monkeypatch.setattr(forecast_actuals, '_SPECIAL_CSV', source)
    missing = attach_actuals([{'date': '2026-10-13', 'prediction': 2000}])
    assert_missing(missing[0])
    source.write_text('date,close\n2026-10-13,2222\n')
    result = attach_actuals([{'date': '2026-10-13', 'prediction': 2000}])
    assert result[0]['actual_price'] == 2222
    assert not yahoo['calls']


@pytest.mark.parametrize('source_contents', [None, 'not_date,not_close\nbad,data\n'])
def test_missing_or_invalid_special_csv_never_falls_back(tmp_path, yahoo, caplog, source_contents):
    source = tmp_path / 'aktual.csv'
    if source_contents is not None:
        source.write_text(source_contents)
    yahoo['frame'] = closes(['2026-10-13'], [9999])
    result = attach_actuals([{'date': '2026-10-13', 'prediction': 2000}], source)
    assert result[0]['prediction'] == 2000
    assert_missing(result[0])
    assert not yahoo['calls']
    assert 'Special actual prices unavailable' in caplog.text


def test_outside_special_range_never_reads_csv(tmp_path, yahoo, monkeypatch):
    def no_csv(*args, **kwargs):
        raise AssertionError('Non-special dates must not read a CSV')
    monkeypatch.setattr(forecast_actuals, '_csv_actuals', no_csv)
    yahoo['frame'] = closes(['2026-10-22'], [2100])
    result = attach_actuals([{'date': '2026-10-22', 'prediction': 2000}], tmp_path / 'any.csv')
    assert result[0]['actual_price'] == 2100


def test_percentage_error_for_overprediction_exact_match_and_invalid_actuals(tmp_path):
    source = tmp_path / 'prices.csv'
    source.write_text('date,close\n2026-10-13,2000\n2026-10-14,2000\n'
                      '2026-10-15,0\n2026-10-16,-1\n2026-10-19,inf\n')
    result = attach_actuals([
        {'date': '2026-10-13', 'prediction': 2020},
        {'date': '2026-10-14', 'prediction': 2000},
        {'date': '2026-10-15', 'prediction': 2010},
        {'date': '2026-10-16', 'prediction': 2010},
        {'date': '2026-10-19', 'prediction': 2010},
    ], source)
    assert result[0]['percentage_error'] == 1
    assert result[1]['absolute_error'] == 0
    assert result[1]['percentage_error'] == 0
    for row in result[2:]:
        assert_missing(row)


@pytest.mark.parametrize('hour,minute,same_day_available', [(16, 59, False), (17, 0, True)])
def test_yahoo_ignores_future_and_unfinished_daily_bars(monkeypatch, yahoo, hour, minute, same_day_available):
    # 21:00 UTC is 17:00 New York during October; use the local-market clock.
    monkeypatch.setattr(forecast_actuals, '_now_new_york', lambda: datetime(
        2026, 10, 29, hour, minute, tzinfo=ZoneInfo('America/New_York')))
    dates = ['2026-10-28', '2026-10-29', '2026-10-30']
    yahoo['frame'] = closes(dates, [2000, 2100, 2200])
    result = attach_actuals([{'date': day, 'prediction': 2020} for day in dates])
    assert result[0]['actual_price'] == 2000
    if same_day_available:
        assert result[1]['actual_price'] == 2100
    else:
        assert_missing(result[1])
    assert_missing(result[2])
    assert len(yahoo['calls']) == 1
    assert yahoo['calls'][0]['end'] == ('2026-10-30' if same_day_available else '2026-10-29')


def test_only_future_and_unclosed_dates_do_not_contact_yahoo(yahoo):
    result = attach_actuals([
        {'date': '2026-10-29', 'prediction': 2000},
        {'date': '2026-10-30', 'prediction': 2010},
    ])
    for row in result:
        assert_missing(row)
    assert not yahoo['calls']


def test_yahoo_missing_weekend_nan_zero_and_negative_are_not_actuals(yahoo):
    dates = ['2026-10-22', '2026-10-23', '2026-10-24', '2026-10-26', '2026-10-27', '2026-10-28']
    yahoo['frame'] = closes(
        ['2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27', '2026-10-28'],
        [2000, float('nan'), 0, -100, float('inf')])
    result = attach_actuals([{'date': day, 'prediction': 2020} for day in dates])
    assert result[0]['actual_price'] == 2000
    for row in result[1:]:
        assert_missing(row)


def test_yahoo_failure_preserves_forecasts(yahoo, caplog):
    yahoo['error'] = RuntimeError('provider unavailable')
    original = [{'date': '2026-10-28', 'prediction': 2020, 'time_idx': 61}]
    result = attach_actuals(original)
    assert result[0]['prediction'] == 2020
    assert result[0]['time_idx'] == 61
    assert_missing(result[0])
    assert original == [{'date': '2026-10-28', 'prediction': 2020, 'time_idx': 61}]
    assert 'Yahoo Finance actual prices unavailable' in caplog.text


@pytest.mark.parametrize('source', ['special_csv', 'yahoo'])
def test_actual_refresh_missing_to_available_without_model(monkeypatch, yahoo, source):
    def no_inference(*args, **kwargs):
        raise AssertionError('Actual refresh must not load or invoke a model')
    for name in ('load_model', 'infer_from_dataframe', 'infer_from_payload', 'infer_tft_model'):
        monkeypatch.setattr(app, name, no_inference)
    client = app.create_app().test_client()
    day = '2026-10-13' if source == 'special_csv' else '2026-10-28'
    original = [{'date': day, 'prediction': 1980, 'raw_prediction': 2200,
                 'time_idx': 61, 'actual_price': 999, 'absolute_error': 999,
                 'percentage_error': 999}]
    first = client.post('/api/forecast-actuals', json={'predictions': original})
    assert first.status_code == 200
    first_rows = first.get_json()['predictions']
    assert_missing(first_rows[0])
    if source == 'special_csv':
        forecast_actuals._SPECIAL_CSV.write_text(f'date,close\n{day},2000\n')
    else:
        yahoo['frame'] = closes([day], [2000])
    second = client.post('/api/forecast-actuals', json={'predictions': first_rows})
    assert second.status_code == 200
    row = second.get_json()['predictions'][0]
    assert row == dict(original[0], actual_price=2000, absolute_error=20, percentage_error=1)
    assert original[0]['actual_price'] == 999


def test_actual_refresh_provider_failure_returns_success_with_nulls(yahoo):
    yahoo['error'] = RuntimeError('offline')
    response = app.create_app().test_client().post('/api/forecast-actuals', json={
        'predictions': [{'date': '2026-10-28', 'prediction': 2000}]})
    assert response.status_code == 200
    assert_missing(response.get_json()['predictions'][0])


@pytest.mark.parametrize('payload', [
    None, [], {}, {'predictions': None}, {'predictions': {}}, {'predictions': []},
    {'predictions': [{'date': '2026-10-28', 'prediction': 2000}] * 8},
    {'predictions': [None]}, {'predictions': [[]]}, {'predictions': [{}]},
    {'predictions': [{'date': '2026-02-30', 'prediction': 2000}]},
    {'predictions': [{'date': '20261028', 'prediction': 2000}]},
    {'predictions': [{'date': '2026-10-28T00:00:00', 'prediction': 2000}]},
    {'predictions': [{'date': 20261028, 'prediction': 2000}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': '2000'}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': True}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': None}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': float('nan')}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': float('inf')}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': -float('inf')}]},
    {'predictions': [{'date': '2026-10-28', 'prediction': 10 ** 400}]},
])
def test_actual_refresh_rejects_malformed_predictions(payload, monkeypatch, yahoo):
    def no_inference(*args, **kwargs):
        raise AssertionError('Malformed refresh must not load a model')
    monkeypatch.setattr(app, 'load_model', no_inference)
    response = app.create_app().test_client().post('/api/forecast-actuals', json=payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()
    assert not yahoo['calls']


@pytest.mark.parametrize('body,content_type', [
    ('{', 'application/json'), ('predictions=[]', 'text/plain'),
])
def test_actual_refresh_rejects_invalid_json(body, content_type, yahoo):
    response = app.create_app().test_client().post(
        '/api/forecast-actuals', data=body, content_type=content_type)
    assert response.status_code == 400
    assert not yahoo['calls']


@pytest.fixture
def coverage_client(monkeypatch):
    def no_external_calls(*args, **kwargs):
        raise AssertionError('Coverage must not contact Yahoo or load/invoke a model')

    for name in ('load_model', 'infer_from_dataframe', 'infer_from_payload', 'infer_tft_model'):
        monkeypatch.setattr(app, name, no_external_calls)
    monkeypatch.setattr(forecast_actuals, '_yahoo_actuals', no_external_calls)
    monkeypatch.setattr(yf, 'Ticker', no_external_calls)
    return app.create_app().test_client()


def test_actuals_coverage_reports_only_exact_positive_finite_observations(coverage_client):
    forecast_actuals._SPECIAL_CSV.write_text(
        'date,close\n'
        '2026-10-22,2200\n'
        '2026-10-21,2100\n'
        '2026-10-13,2000\n'
        '2026-10-12,1900\n'
        '2026-10-16,2050\n'
        '2026-10-14,NaN\n'
        '2026-10-15,0\n'
        '2026-10-19,-1\n'
        '2026-10-20,inf\n'
        '2026-10-13,NaN\n'
        '2026-10-14T00:00:00,2300\n'
        '20261015,2300\n'
        '2026-02-30,2300\n'
        '2026-9-01,2300\n')
    response = coverage_client.get('/api/actuals-coverage')
    assert response.status_code == 200
    assert response.get_json() == {
        'special_start': '2026-10-13',
        'special_end': '2026-10-21',
        'csv': {
            'available': True,
            'first_date': '2026-10-12',
            'last_date': '2026-10-22',
            'special_dates': ['2026-10-13', '2026-10-16', '2026-10-21'],
        },
    }


def test_actuals_coverage_distinguishes_history_ending_before_special_interval(coverage_client):
    forecast_actuals._SPECIAL_CSV.write_text(
        'date,close\n2026-10-09,1900\n2026-10-12,2000\n2026-10-13,NaN\n')
    response = coverage_client.get('/api/actuals-coverage')
    assert response.status_code == 200
    assert response.get_json() == {
        'special_start': '2026-10-13',
        'special_end': '2026-10-21',
        'csv': {
            'available': True,
            'first_date': '2026-10-09',
            'last_date': '2026-10-12',
            'special_dates': [],
        },
    }


@pytest.mark.parametrize('contents', [
    None,
    b'',
    b'not_date,not_close\nbad,data\n',
    b'date,close\n2026-10-13,\xff\n',
    b'date,close\n2026-10-13,NaN\n2026-10-14,0\ninvalid,2000\n',
])
def test_missing_or_invalid_coverage_csv_keeps_page_available(
        coverage_client, contents, caplog):
    source = forecast_actuals._SPECIAL_CSV
    if contents is not None:
        source.write_bytes(contents)
    response = coverage_client.get('/api/actuals-coverage')
    assert response.status_code == 200
    assert response.get_json() == {
        'special_start': '2026-10-13',
        'special_end': '2026-10-21',
        'csv': {
            'available': False,
            'first_date': None,
            'last_date': None,
            'special_dates': [],
        },
    }
    assert str(source.parent) not in response.get_data(as_text=True)
    assert str(source.parent) not in caplog.text
    assert coverage_client.get('/').status_code == 200


def test_actuals_coverage_rereads_csv_without_generating_forecast(coverage_client):
    source = forecast_actuals._SPECIAL_CSV
    source.write_text('date,close\n2026-10-12,1900\n')
    first = coverage_client.get('/api/actuals-coverage')
    assert first.status_code == 200
    assert first.get_json()['csv'] == {
        'available': True, 'first_date': '2026-10-12',
        'last_date': '2026-10-12', 'special_dates': [],
    }
    source.write_text('date,close\n2026-10-12,1900\n2026-10-13,2000\n')
    second = coverage_client.get('/api/actuals-coverage')
    assert second.status_code == 200
    assert second.get_json()['csv'] == {
        'available': True, 'first_date': '2026-10-12',
        'last_date': '2026-10-13', 'special_dates': ['2026-10-13'],
    }
