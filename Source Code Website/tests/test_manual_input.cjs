const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map(match => match[1]).join('\n');

function page(options = {}) {
    const values = {
        goldPrice: '2350', usdIndex: '104', sp500: '5200', oilPrice: '78',
        fedRate: '5.25',
        futureUsd: '104', futureSp500: '5200', futureOil: '78'
    };
    const elements = new Map();
    function element(id) {
        if (!elements.has(id)) {
            let inputValue = values[id] || '';
            const node = {style: {}, classList: {add() {}, remove() {}}, addEventListener() {}};
            Object.defineProperty(node, 'value', {get: () => inputValue, set: value => {inputValue = String(value);}});
            elements.set(id, node);
        }
        return elements.get(id);
    }
    const history = Array.from({length: 100}, (_, i) => ({
        date: new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10),
        close: 2200 + i, open: 2199 + i, tickvol: 50000 + i,
        dxy_open: 99 + i / 100, dxy_close: 100 + i / 100, fed_dff_pct: 4.25,
        usd_close: 100 + i / 100, usd_open: 99 + i / 100, usd_adj_close: 100 + i / 100,
        gspc_close: 5000 + i, gspc_open: 4999 + i, oil_dcoilwtico: 70 + i / 10,
        fed_rate_pct: 4.25, fed_target_rate_from_pct: 4.25,
        fed_target_rate_to_pct: 4.5, cpi_cpiaucsl: 320
    }));
    const requests = [];
    const downloads = [];
    const actualRequests = [];
    const coverageRequests = [];
    const charts = [];
    const listeners = new Map();
    const intervals = [];
    const storage = options.storage || new Map();
    const document = {
        hidden: false,
        addEventListener(type, callback) {
            if (!listeners.has(type)) listeners.set(type, []);
            listeners.get(type).push(callback);
        },
        getElementById: element, querySelector: element, querySelectorAll: () => [],
        createElement: () => ({click() {}})
    };
    let scenarioCalls = 0;
    const pageOptions = options;
    const context = vm.createContext({
        console,
        document,
        setInterval(callback, delay) { intervals.push({callback, delay}); },
        localStorage: {
            getItem(key) {
                if (options.storageUnavailable) throw new Error('Storage unavailable');
                return storage.get(key) ?? null;
            },
            setItem(key, value) {
                if (options.storageUnavailable) throw new Error('Storage unavailable');
                storage.set(key, value);
            }
        },
        Plotly: { purge() {}, newPlot(id, traces, layout) { charts.push({id, traces, layout}); } },
        Blob: class { constructor(parts) { downloads.push(parts.join('')); } },
        URL: {createObjectURL: () => 'blob:test'},
        fetch: async (url, options) => {
            if (url === '/status') return {ok: true, json: async () => ({models: ['TFT']})};
            if (url === '/api/actuals-coverage') {
                coverageRequests.push(url);
                return {ok: true, json: async () => ({
                    special_start: '2026-10-13', special_end: '2026-10-21',
                    csv: {available: true, first_date: '2020-01-01',
                        last_date: '2026-10-12', special_dates: []}
                })};
            }
            if (url === '/api/forecast-actuals') {
                const body = JSON.parse(options.body);
                actualRequests.push(body);
                const predictions = pageOptions.refreshActuals
                    ? await pageOptions.refreshActuals(body.predictions, actualRequests.length)
                    : body.predictions.map(row => ({
                        ...row, actual_price: null, absolute_error: null, percentage_error: null
                    }));
                return {ok: true, json: async () => ({predictions})};
            }
            if (url === '/load-dataset') {
                if (pageOptions.datasetGate) await pageOptions.datasetGate;
                return {ok: true, json: async () => ({success: true, source: 'Dataset CSV',
                    data: history.map(row => ({...row, close: row.close + 2000})),
                    date_range: [history[0].date, history.at(-1).date]})};
            }
            if (url === '/sample-scenarios') {
                scenarioCalls++;
                if (pageOptions.scenarioGate) await pageOptions.scenarioGate;
                return {ok: !pageOptions.scenarioError, json: async () => pageOptions.scenarioError
                    ? {error: 'Source unavailable'}
                    : {success: true, source: 'Actual CSV fixture', scenarios: {
                        bullish: {name: 'Bullish', description: 'Historical uptrend', data: history.slice(-60).map(row => ({...row, close: row.close + 1000})), history: history.map(row => ({...row, close: row.close + 1000}))}
                    }, unavailable: {bearish: 'No historical bearish window'}}};
            }
            if (url === '/load-dataset' && pageOptions.datasetError) throw new Error('Missing dataset');
            assert.equal(url, '/predict', 'manual mode must not fetch API history');
            requests.push(JSON.parse(options.body));
            if (pageOptions.predictionGate) await pageOptions.predictionGate;
            if (pageOptions.predictions) return {
                ok: true, json: async () => ({mode: 'TFT-model', predictions: pageOptions.predictions})
            };
            return { ok: false, json: async () => ({ error: 'Test capture' }) };
        }
    });
    vm.runInContext(script, context);
    vm.runInContext("currentMode = 'manual'", context);
    const csv = [Object.keys(history[0]).join(','), ...history.map(row => Object.values(row).join(','))].join('\n');
    return {context, element, requests, actualRequests, coverageRequests, charts, storage, history, downloads,
        intervals, document,
        initialize: () => { for (const callback of listeners.get('DOMContentLoaded') || []) callback(); },
        visibility: hidden => {
            document.hidden = hidden;
            for (const callback of listeners.get('visibilitychange') || []) callback();
        },
        tick: () => Promise.all(intervals.map(timer => timer.callback())),
        scenarioCalls: () => scenarioCalls,
        switchTab: tab => {
            context.newTab = tab;
            return vm.runInContext('switchInputTab(newTab)', context);
        },
        loadDataset: () => vm.runInContext("loadFromDataset({currentTarget: document.getElementById('datasetButton')})", context),
        uploadCSV: () => {
            context.uploadFile = {name: 'actual.csv', text: async () => csv};
            vm.runInContext("currentMode = 'upload'", context);
            return vm.runInContext('handleFileUpload(uploadFile)', context);
        },
        upload: (rows = history) => {
            context.uploadHistory = rows;
            vm.runInContext("currentMode = 'upload'; setSourceHistory('upload', uploadHistory); configureForecastPeriod(true)", context);
        },
        cutoff: date => {
            element('forecastCutoff').value = date;
            return vm.runInContext('updateForecastPeriod()', context);
        },
        scenario: type => {
            context.selectedScenario = type;
            return vm.runInContext('loadScenario(selectedScenario)', context);
        },
        load: (text = csv) => {
            context.csvFile = {name: 'actual.csv', text: async () => text};
            return vm.runInContext('handleManualCSV({files: [csvFile]})', context);
        },
        run: () => vm.runInContext('runPrediction()', context)};
}

test('repeated manual forecasts reuse the same actual history', async () => {
    const app = page();
    await app.load();
    await app.run();
    await app.run();
    assert.equal(app.requests.length, 2);
    assert.deepEqual(app.requests[0], app.requests[1]);
    assert.equal(app.requests[0].input_mode, 'manual');
});

test('manual edits only the latest observation and preserves actual historical OHLC and volume', async () => {
    const app = page();
    await app.load();
    app.element('goldPrice').value = '2350';
    app.element('usdIndex').value = '104';
    app.element('fedRate').value = '5.25';
    await app.run();
    const history = app.requests[0].historical_data;
    assert.equal(history.length, 100);
    assert.deepEqual(history.slice(0, -1), app.history.slice(0, -1));
    assert.equal(history.at(-1).close, 2350);
    assert.equal(history.at(-1).dxy_close, 104);
    assert.equal(history.at(-1).open, app.history.at(-1).open);
    assert.equal(history.at(-1).tickvol, app.history.at(-1).tickvol);
    assert.equal(history.at(-1).fed_dff_pct, 5.25);
    assert.equal(app.history.at(-1).close, 2299);
});

test('manual mode loads the bundled actual history automatically', async () => {
    const app = page();
    await app.run();
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].historical_data.length, 100);
    assert.equal(app.element('btnPredict').disabled, false);
});

test('empty manual values prevent prediction instead of inserting defaults', async () => {
    const app = page();
    await app.load();
    app.element('usdIndex').value = '';
    await app.run();
    assert.equal(app.requests.length, 0);
});

test('changing a manual field changes the payload and zero Fed Rate is preserved', async () => {
    const app = page();
    await app.load();
    await app.run();
    app.element('usdIndex').value = '106';
    app.element('fedRate').value = '0';
    await app.run();
    assert.notDeepEqual(app.requests[0], app.requests[1]);
    assert.equal(app.requests[1].historical_data.at(-1).dxy_close, 106);
    assert.equal(app.requests[1].historical_data.at(-1).fed_dff_pct, 0);
    assert.equal('future_7' in app.requests[1], false);
});

test('manual forecasts use future form values after previously loading API data', async () => {
    const app = page();
    await app.load();
    vm.runInContext('apiFutureData = Array(7).fill({usd_close: 999})', app.context);
    await app.run();
    assert.equal('future_7' in app.requests[0], false);
    assert.equal(app.requests[0].historical_data.at(-1).usd_close, 100.99);
});

test('incomplete or invalid CSV is rejected without inventing missing values', async () => {
    const app = page();
    await app.load('date,close\n2026-01-01,2300');
    assert.match(app.element('statusMessage').className, /error/);
    assert.equal(app.requests.length, 0);
});

test('invalid replacement CSV clears previously loaded history', async () => {
    const app = page();
    await app.load();
    await app.load('date,close\n2026-01-01,2300');
    assert.equal(app.requests.length, 0);
});

test('quoted CSV preserves observations and rejects duplicate dates', async () => {
    const app = page();
    const headers = Object.keys(app.history[0]);
    const quoted = [headers, ...app.history.map(row => headers.map(key => row[key]))]
        .map(row => row.map(value => `"${value}"`).join(',')).join('\r\n');
    await app.load(quoted);
    await app.run();
    assert.equal(app.requests.length, 1);
    app.history[0].date = app.history[1].date;
    await app.load([headers.join(','), ...app.history.map(row => headers.map(key => row[key]).join(','))].join('\n'));
    await app.run();
    assert.equal(app.requests.length, 1);
});

test('CSV export preserves the forecast observation date in Asia/Jakarta', () => {
    const oldTimezone = process.env.TZ;
    process.env.TZ = 'Asia/Jakarta';
    try {
        const app = page();
        vm.runInContext("currentPredictions = [{date: '2026-03-02', prediction: 2300}]; downloadCSV()", app.context);
        assert.match(app.downloads[0], /1,2026-03-02,2300\.0000,/);
    } finally {
        if (oldTimezone === undefined) delete process.env.TZ;
        else process.env.TZ = oldTimezone;
    }
});

test('upload preserves CSV observations and ignores unsupported future parameters', async () => {
    const app = page();
    await app.uploadCSV();

    await app.run();
    app.element('futureUsd').value = '120';
    app.element('futureSp500').value = '6000';
    app.element('futureOil').value = '110';
    vm.runInContext('apiFutureData = Array(7).fill({usd_close: 999})', app.context);
    await app.run();
    assert.equal(app.requests[1].historical_data.at(-1).usd_close, 100.99);
    assert.equal(app.requests[1].historical_data.at(-1).gspc_close, 5099);
    assert.equal(app.requests[1].historical_data.at(-1).oil_dcoilwtico, 79.9);
    assert.deepEqual(app.requests[0], app.requests[1]);
    assert.deepEqual(app.requests[1].historical_data.slice(0, -1), app.requests[0].historical_data.slice(0, -1));
    assert.equal(app.history.at(-1).usd_close, 100.99);
});

test('sample clicks reuse the same selected CSV window', async () => {
    const app = page();
    await app.scenario('bullish');
    await app.run();
    await app.scenario('bullish');
    await app.run();
    assert.equal(app.requests.length, 2);
    assert.deepEqual(app.requests[0], app.requests[1]);
    assert.equal(app.scenarioCalls(), 1);
});

test('missing scenario clears previous data and never generates random history', async () => {
    const app = page();
    await app.scenario('bullish');
    await app.scenario('bearish');
    assert.match(app.element('statusMessage').className, /error/);
    assert.equal(app.requests.length, 0);
});

test('scenario source error never creates random history', async () => {
    const app = page({scenarioError: true});
    await app.scenario('bullish');
    await app.run();
    assert.equal(app.requests.length, 0);
});

test('switching Sample to Upload cannot forecast with scenario history', async () => {
    const app = page();
    await app.scenario('bullish');
    app.switchTab('upload');
    await app.run();
    assert.equal(app.requests.length, 0);
});

test('late scenario response cannot replace an uploaded CSV', async () => {
    let release;
    const scenarioGate = new Promise(resolve => {release = resolve;});
    const app = page({scenarioGate});
    const pending = app.scenario('bullish');
    app.switchTab('upload');
    await app.uploadCSV();
    release();
    await pending;
    await app.run();
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].historical_data.at(-1).close, 2299);
});

test('small TFT changes remain visible and CSV matches displayed precision', () => {
    const app = page();
    vm.runInContext("renderTable([{date: '2024-02-26', prediction: 2044.79150390625}])", app.context);
    const first = app.element('#resultsTable tbody').innerHTML;
    vm.runInContext("renderTable([{date: '2024-02-26', prediction: 2044.785888671875}])", app.context);
    const second = app.element('#resultsTable tbody').innerHTML;
    assert.notEqual(first, second);
    assert.match(first, /2044\.7915/);
    assert.match(second, /2044\.7859/);
    vm.runInContext("currentPredictions = [{date: '2024-02-26', prediction: 2044.79150390625}]; downloadCSV()", app.context);
    assert.equal(app.downloads[0].trimEnd().split('\n')[1], '1,2024-02-26,2044.7915,,,');
});

test('forecast table and CSV include percentage error and preserve unavailable actuals', () => {
    const app = page();
    vm.runInContext(`currentPredictions = [
        {date: '2026-02-19', prediction: 1980, actual_price: 2000, absolute_error: 20, percentage_error: 1},
        {date: '2026-02-20', prediction: 2000, actual_price: 2000, absolute_error: 0, percentage_error: 0},
        {date: '2026-02-21', prediction: 2010, actual_price: null, absolute_error: null, percentage_error: null}
    ]; renderTable(currentPredictions); downloadCSV()`, app.context);
    const table = app.element('#resultsTable tbody').innerHTML;
    assert.match(table, /<td>1\.00%<\/td>/);
    assert.match(table, /<td>0\.00%<\/td>/);
    const lines = app.downloads[0].trimEnd().split('\n');
    assert.equal(lines[0].split(',').at(-1), 'Absolute Percentage Error (%)');
    assert.equal(lines[1], '1,2026-02-19,1980.0000,2000.0000,20.0000,1.00');
    assert.equal(lines[2], '2,2026-02-20,2000.0000,2000.0000,0.0000,0.00');
    assert.equal(lines[3], '3,2026-02-21,2010.0000,,,');
});

test('late dataset response cannot replace the CSV selected in Upload', async () => {
    let release;
    const datasetGate = new Promise(resolve => {release = resolve;});
    const app = page({datasetGate});
    app.switchTab('dataset');
    const pending = app.loadDataset();
    app.switchTab('upload');
    await app.uploadCSV();
    release();
    await pending;
    await app.run();
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].input_mode, 'upload');
    assert.equal(app.requests[0].historical_data.at(-1).close, 2299);
});

function forecastRows(base = 2300) {
    return Array.from({length: 7}, (_, i) => ({
        date: `2026-10-${String(13 + i).padStart(2, '0')}`,
        prediction: base + i / 10,
        actual_price: null, absolute_error: null, percentage_error: null
    }));
}

function storedForecast(predictions = forecastRows()) {
    return new Map([['tft:last-forecast', JSON.stringify({mode: 'TFT-model', predictions})]]);
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('successful TFT forecast survives reload and refreshes actuals without changing its chart or summary', async () => {
    const storage = new Map();
    const generated = page({storage, predictions: forecastRows()});
    await generated.run();
    assert.equal(generated.requests.length, 1);
    assert.equal(storage.size, 1);
    let release;
    const gate = new Promise(resolve => {release = resolve;});
    const restored = page({storage, refreshActuals: async predictions => {
        await gate;
        return predictions.map(row => ({
            ...row, actual_price: 2400,
            absolute_error: Math.abs(row.prediction - 2400),
            percentage_error: Math.abs(row.prediction - 2400) / 24
        }));
    }});
    restored.initialize();
    const initialTable = restored.element('#resultsTable tbody').innerHTML;
    assert.match(initialTable, /Oct 13, 2026/);
    assert.equal((initialTable.match(/<td>-<\/td>/g) || []).length, 21);
    assert.match(initialTable, /\$2300\.0000/);
    assert.equal(restored.element('modelBadge').style.display, 'inline-flex');
    assert.match(restored.element('modelBadge').textContent, /TFT Model/);
    const summary = restored.element('forecastSummary').innerHTML;
    const chart = JSON.stringify(restored.charts);
    assert.match(summary, /\$2300\.6000/);
    assert.equal(restored.charts.length, 1);
    assert.equal(restored.charts[0].traces[0].y[0], 2300);
    release();
    await settle();
    const updated = restored.element('#resultsTable tbody').innerHTML;
    assert.match(updated, /\$2400\.0000/);
    assert.match(updated, /\$100\.0000/);
    assert.match(updated, /4\.17%/);
    assert.equal(restored.element('forecastSummary').innerHTML, summary);
    assert.equal(JSON.stringify(restored.charts), chart);
    assert.equal(restored.requests.length, 0);
    const saved = JSON.parse(storage.get('tft:last-forecast')).predictions;
    assert.deepEqual(saved.map(row => row.prediction), forecastRows().map(row => row.prediction));
    assert.equal(saved[0].actual_price, 2400);
});

test('actual refresh polls each minute only while visible and never overlaps a pending request', async () => {
    let release;
    const gate = new Promise(resolve => {release = resolve;});
    const app = page({storage: storedForecast(), refreshActuals: async (predictions, call) => {
        if (call === 1) await gate;
        return predictions.map(row => ({
            ...row, actual_price: 2400 + call, absolute_error: 100 + call, percentage_error: call
        }));
    }});
    app.initialize();
    assert.deepEqual(app.intervals.map(timer => timer.delay), [60000]);
    await app.tick();
    app.visibility(false);
    assert.equal(app.actualRequests.length, 1);
    release();
    await settle();
    assert.match(app.element('#resultsTable tbody').innerHTML, /\$2401\.0000/);
    app.visibility(true);
    await app.tick();
    assert.equal(app.actualRequests.length, 1);
    app.visibility(false);
    await settle();
    assert.match(app.element('#resultsTable tbody').innerHTML, /\$2402\.0000/);
    await app.tick();
    assert.match(app.element('#resultsTable tbody').innerHTML, /\$2403\.0000/);
    assert.equal(app.requests.length, 0);
});

test('a late actual response cannot resurrect a forecast cleared by changing input mode', async () => {
    let release;
    const gate = new Promise(resolve => {release = resolve;});
    const app = page({storage: storedForecast(), refreshActuals: async predictions => {
        await gate;
        return predictions.map(row => ({
            ...row, actual_price: 9999, absolute_error: 1, percentage_error: 1
        }));
    }});
    app.initialize();
    app.switchTab('upload');
    const cleared = app.element('#resultsTable tbody').innerHTML;
    release();
    await settle();
    assert.equal(app.element('#resultsTable tbody').innerHTML, cleared);
    assert.equal(app.element('modelBadge').style.display, 'none');
    assert.doesNotMatch(cleared, /2300|9999/);
    assert.equal(app.requests.length, 0);
});

test('a late actual response cannot overwrite a newly generated TFT forecast or its storage', async () => {
    let release;
    const gate = new Promise(resolve => {release = resolve;});
    const app = page({storage: storedForecast(), predictions: forecastRows(3000),
        refreshActuals: async predictions => {
            await gate;
            return predictions.map(row => ({
                ...row, actual_price: 9999, absolute_error: 1, percentage_error: 1
            }));
        }});
    app.initialize();
    await app.run();
    const newTable = app.element('#resultsTable tbody').innerHTML;
    const newStorage = app.storage.get('tft:last-forecast');
    release();
    await settle();
    assert.match(newTable, /\$3000\.0000/);
    assert.equal(app.element('#resultsTable tbody').innerHTML, newTable);
    assert.equal(app.storage.get('tft:last-forecast'), newStorage);
    assert.equal(app.requests.length, 1);
});

test('unavailable storage does not prevent successful forecast rendering', async () => {
    const app = page({storageUnavailable: true, predictions: forecastRows()});
    app.initialize();
    await app.run();
    assert.match(app.element('#resultsTable tbody').innerHTML, /\$2300\.0000/);
    assert.equal(app.element('modelBadge').style.display, 'inline-flex');
    assert.equal(app.charts.length, 1);
    assert.equal(app.requests.length, 1);
});

test('malformed or non-TFT stored forecasts are not restored or refreshed', async () => {
    const invalidDate = forecastRows();
    invalidDate[0].date = '2026-02-30';
    const invalidPrice = forecastRows();
    invalidPrice[0].prediction = '2300';
    const invalidValues = [
        'not json', 'null', '{}',
        JSON.stringify({mode: 'demo', predictions: forecastRows()}),
        JSON.stringify({mode: 'TFT-model', predictions: forecastRows().slice(1)}),
        JSON.stringify({mode: 'TFT-model', predictions: invalidDate}),
        JSON.stringify({mode: 'TFT-model', predictions: invalidPrice})
    ];
    for (const value of invalidValues) {
        const app = page({storage: new Map([['tft:last-forecast', value]])});
        app.initialize();
        await settle();
        assert.equal(app.actualRequests.length, 0);
        assert.equal(app.charts.length, 0);
        assert.equal(app.requests.length, 0);
    }
});

test('failed or mismatched actual refresh preserves the visible forecast and saved values', async () => {
    const predictions = forecastRows().map(row => ({
        ...row, actual_price: 2400, absolute_error: 100, percentage_error: 4.17
    }));
    const app = page({storage: storedForecast(predictions), refreshActuals: async (rows, call) => {
        if (call === 1) throw new Error('Offline');
        return rows.map(row => ({
            ...row, prediction: 0, actual_price: 9999, absolute_error: 1, percentage_error: 1
        }));
    }});
    app.initialize();
    const initialTable = app.element('#resultsTable tbody').innerHTML;
    const saved = app.storage.get('tft:last-forecast');
    await settle();
    assert.equal(app.element('#resultsTable tbody').innerHTML, initialTable);
    await app.tick();
    assert.equal(app.element('#resultsTable tbody').innerHTML, initialTable);
    assert.equal(app.storage.get('tft:last-forecast'), saved);
    assert.equal(app.requests.length, 0);
});

test('forecast cutoff is unavailable without history and follows the loaded raw observation range', async () => {
    const app = page();
    await app.switchTab('upload');
    assert.equal(app.element('forecastCutoff').disabled, true);
    await app.uploadCSV();
    const picker = app.element('forecastCutoff');
    assert.equal(picker.disabled, false);
    assert.equal(picker.min, '2026-03-21');
    assert.equal(picker.max, '2026-04-10');
    assert.equal(picker.value, '2026-04-10');
    assert.equal(app.requests.length, 0);
});

for (const source of ['upload', 'dataset']) {
    test(`${source} sends only the selected actual history through the cutoff`, async () => {
        const app = page();
        if (source === 'upload') {
            await app.uploadCSV();
        } else {
            await app.switchTab('dataset');
            await app.loadDataset();
        }
        app.cutoff('2026-03-25');
        await app.run();
        assert.equal(app.requests.length, 1);
        const request = app.requests[0];
        const expected = app.history.slice(0, 84).map(row => ({
            ...row, close: row.close + (source === 'dataset' ? 2000 : 0)
        }));
        assert.equal(request.input_mode, source);
        assert.equal(request.historical_data.length, 84);
        assert.deepEqual(request.historical_data, expected);
        assert.equal(request.historical_data.at(-1).date, '2026-03-25');
        assert.ok(request.historical_data.every(row => row.date <= '2026-03-25'));
        app.cutoff('2026-04-10');
        await app.run();
        assert.equal(app.requests[1].historical_data.length, 100);
        assert.equal(app.requests[1].historical_data.at(-1).date, '2026-04-10');
    });
}

test('the 80-observation cutoff boundary is allowed but an earlier date sends no prediction', async () => {
    const app = page();
    app.upload();
    app.cutoff('2026-03-20');
    await app.run();
    assert.equal(app.requests.length, 0);
    assert.match(app.element('statusMessage').className, /error/);
    app.cutoff('2026-03-21');
    await app.run();
    assert.equal(app.requests.length, 1);
    assert.deepEqual(app.requests[0].historical_data, app.history.slice(0, 80));
    assert.equal(app.requests[0].historical_data.at(-1).date, '2026-03-21');
});

for (const cutoff of ['2026-04-11', '', '2026-02-30', '03/25/2026']) {
    test(`invalid or unavailable cutoff ${JSON.stringify(cutoff)} cannot silently use the latest history`, async () => {
        const app = page();
        app.upload();
        app.cutoff(cutoff);
        await app.run();
        assert.equal(app.requests.length, 0);
        assert.match(app.element('statusMessage').className, /error/);
    });
}

test('a missing observation is not padded and the preview uses the last real observation', async () => {
    const app = page();
    const rows = app.history.filter(row => row.date !== '2026-03-25');
    app.upload(rows);
    app.cutoff('2026-03-25');
    await app.run();
    assert.equal(app.requests.length, 1);
    const selected = app.requests[0].historical_data;
    assert.equal(selected.length, 83);
    assert.deepEqual(selected, rows.filter(row => row.date <= '2026-03-25'));
    assert.equal(selected.at(-1).date, '2026-03-24');
    const preview = app.element('forecastPeriodPreview').textContent;
    assert.match(preview, /2026-03-24/);
    assert.match(preview, /2026-03-25/);
    assert.match(preview, /2026-04-02/);
});

test('weekend cutoff planning starts seven weekdays after the effective Friday observation', async () => {
    const app = page();
    const dates = [];
    const cursor = new Date('2026-04-10T00:00:00Z');
    while (dates.length < 100) {
        if (cursor.getUTCDay() !== 0 && cursor.getUTCDay() !== 6) {
            dates.unshift(cursor.toISOString().slice(0, 10));
        }
        cursor.setUTCDate(cursor.getUTCDate() - 1);
    }
    const rows = app.history.map((row, i) => ({...row, date: dates[i]}));
    app.upload(rows);
    for (const cutoff of ['2026-03-28', '2026-03-29']) {
        app.cutoff(cutoff);
        const preview = app.element('forecastPeriodPreview').textContent;
        assert.match(preview, /2026-03-27/);
        assert.match(preview, /2026-03-30/);
        assert.match(preview, /2026-04-07/);
        await app.run();
        const selected = app.requests.at(-1).historical_data;
        assert.deepEqual(selected, rows.filter(row => row.date <= cutoff));
        assert.equal(selected.at(-1).date, '2026-03-27');
        assert.ok(selected.every(row => ![0, 6].includes(new Date(row.date + 'T00:00:00Z').getUTCDay())));
    }
    assert.deepEqual(app.requests[0], app.requests[1]);
});

test('manual cutoff resets fields to its selected endpoint and overrides only a copy of that endpoint', async () => {
    const app = page();
    await app.load();
    app.element('goldPrice').value = '9999';
    app.element('usdIndex').value = '999';
    app.cutoff('2026-03-25');
    const endpoint = app.history[83];
    assert.equal(Number(app.element('goldPrice').value), endpoint.close);
    assert.equal(Number(app.element('usdIndex').value), endpoint.dxy_close);
    assert.equal(Number(app.element('sp500').value), endpoint.gspc_close);
    assert.equal(Number(app.element('oilPrice').value), endpoint.oil_dcoilwtico);
    app.element('goldPrice').value = '2450';
    app.element('usdIndex').value = '107';
    app.element('fedRate').value = '0';
    await app.run();
    assert.equal(app.requests.length, 1);
    const selected = app.requests[0].historical_data;
    assert.equal(selected.length, 84);
    assert.deepEqual(selected.slice(0, -1), app.history.slice(0, 83));
    assert.deepEqual(selected.at(-1), {...endpoint, close: 2450, dxy_close: 107, fed_dff_pct: 0});
    assert.ok(selected.every(row => row.date <= '2026-03-25'));
    app.cutoff('2026-04-10');
    assert.equal(Number(app.element('goldPrice').value), app.history.at(-1).close);
    await app.run();
    assert.deepEqual(app.requests[1].historical_data, app.history);
    app.cutoff('2026-03-25');
    assert.equal(Number(app.element('goldPrice').value), endpoint.close);
    await app.run();
    assert.deepEqual(app.requests[2].historical_data, app.history.slice(0, 84));
});

test('source switches retain separate cutoffs and reloading a source resets only that source to latest', async () => {
    const app = page();
    await app.uploadCSV();
    app.cutoff('2026-03-25');
    await app.switchTab('dataset');
    assert.equal(app.element('forecastCutoff').disabled, true);
    await app.loadDataset();
    assert.equal(app.element('forecastCutoff').value, '2026-04-10');
    app.cutoff('2026-03-30');
    await app.run();
    assert.equal(app.requests[0].historical_data.length, 89);
    assert.equal(app.requests[0].historical_data.at(-1).close, app.history[88].close + 2000);
    await app.switchTab('upload');
    assert.equal(app.element('forecastCutoff').value, '2026-03-25');
    await app.run();
    assert.deepEqual(app.requests[1].historical_data, app.history.slice(0, 84));
    await app.uploadCSV();
    assert.equal(app.element('forecastCutoff').value, '2026-04-10');
    await app.run();
    assert.deepEqual(app.requests[2].historical_data, app.history);
    await app.switchTab('dataset');
    assert.equal(app.element('forecastCutoff').value, '2026-03-30');
    await app.run();
    assert.equal(app.requests[3].historical_data.length, 89);
    assert.equal(app.requests[3].historical_data.at(-1).date, '2026-03-30');
});

test('changing cutoff invalidates a pending generated forecast instead of rendering or saving its stale dates', async () => {
    let release;
    const predictionGate = new Promise(resolve => {release = resolve;});
    const dates = ['2026-03-26', '2026-03-27', '2026-03-30', '2026-03-31',
        '2026-04-01', '2026-04-02', '2026-04-03'];
    const app = page({predictionGate, predictions: dates.map(date => ({date, prediction: 3210}))});
    app.upload();
    app.cutoff('2026-03-25');
    const pending = app.run();
    await settle();
    assert.equal(app.requests.length, 1);
    assert.deepEqual(app.requests[0].historical_data, app.history.slice(0, 84));
    app.cutoff('2026-03-26');
    const clearedTable = app.element('#resultsTable tbody').innerHTML;
    release();
    await pending;
    assert.equal(app.element('forecastCutoff').value, '2026-03-26');
    assert.equal(app.element('#resultsTable tbody').innerHTML, clearedTable);
    assert.doesNotMatch(clearedTable, /3210|Mar 26, 2026/);
    assert.equal(app.element('modelBadge').style.display, 'none');
    assert.equal(app.charts.length, 0);
    assert.equal(app.storage.has('tft:last-forecast'), false);
    assert.equal(app.element('btnPredict').disabled, false);
});

test('planning a new cutoff clears visible results without overwriting the persisted forecast', async () => {
    const app = page({storage: storedForecast()});
    app.initialize();
    await settle();
    const saved = app.storage.get('tft:last-forecast');
    app.upload();
    app.cutoff('2026-03-25');
    assert.doesNotMatch(app.element('#resultsTable tbody').innerHTML, /2300\.0000/);
    assert.equal(app.element('modelBadge').style.display, 'none');
    assert.equal(app.storage.get('tft:last-forecast'), saved);
    assert.equal(app.requests.length, 0);
});

test('coverage warns that the special forecast dates are missing from the CSV and refreshes before Generate', async () => {
    const app = page();
    app.initialize();
    await settle();
    assert.equal(app.coverageRequests.length, 1);
    const rows = app.history.map((row, i) => ({
        ...row, date: new Date(Date.UTC(2026, 6, 5 + i)).toISOString().slice(0, 10)
    }));
    app.upload(rows);
    assert.equal(app.element('forecastCutoff').value, '2026-10-12');
    const preview = app.element('forecastPeriodPreview').textContent;
    assert.match(preview, /2026-10-13/);
    assert.match(preview, /2026-10-21/);
    const coverage = app.element('forecastActualCoverage').textContent;
    assert.match(coverage, /CSV/i);
    assert.match(coverage, /2026-10-12/);
    assert.match(coverage, /missing|unavailable|not available|tidak|belum|kosong/i);
    assert.match(coverage, /2026-10-13/);
    app.cutoff('2026-10-11');
    assert.equal(app.coverageRequests.length, 1, 'date planning uses cached coverage');
    app.cutoff('2026-10-12');
    await app.run();
    assert.equal(app.coverageRequests.length, 2, 'Generate refreshes CSV availability');
    assert.equal(app.requests.length, 1, 'missing actuals do not block generating a forecast');
    assert.deepEqual(app.requests[0].historical_data, rows);
});

test('manual and uploaded history keep independent cutoff choices when switching modes', async () => {
    const app = page();
    await app.load();
    app.cutoff('2026-03-25');
    await app.uploadCSV();
    app.cutoff('2026-03-30');
    await app.switchTab('manual');
    assert.equal(app.element('forecastCutoff').value, '2026-03-25');
    await app.run();
    assert.equal(app.requests[0].input_mode, 'manual');
    assert.deepEqual(app.requests[0].historical_data, app.history.slice(0, 84));
    await app.switchTab('upload');
    assert.equal(app.element('forecastCutoff').value, '2026-03-30');
    await app.run();
    assert.equal(app.requests[1].input_mode, 'upload');
    assert.deepEqual(app.requests[1].historical_data, app.history.slice(0, 89));
});

test('ordinary forecast dates explain Yahoo availability rather than implying CSV coverage', async () => {
    const app = page();
    app.initialize();
    await settle();
    app.upload();
    app.cutoff('2026-03-25');
    const coverage = app.element('forecastActualCoverage').textContent;
    assert.match(coverage, /Yahoo/i);
    assert.match(coverage, /check|generation|forecast|periksa|pembaruan/i);
    assert.doesNotMatch(coverage, /2026-10-12/);
    assert.equal(app.coverageRequests.length, 1);
    assert.equal(app.requests.length, 0);
});
