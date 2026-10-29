// Front-end logic for TFT Prediction

async function loadVersions() {
  try {
    const resp = await fetch('/versions');
    const data = await resp.json();
    const select = document.getElementById('version');
    select.innerHTML = '';
    (data.versions || []).forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load versions', e);
  }
}

async function handlePredict() {
  const v = document.getElementById('version').value;
  const jsonData = document.getElementById('jsonData').value;
  const status = document.getElementById('status');
  
  status.textContent = 'Memproses...';
  status.className = '';
  
  try {
    const data = JSON.parse(jsonData);
    
    // Validasi input
    if (!data.encoder_last_60 || data.encoder_last_60.length !== 60) {
      throw new Error('encoder_last_60 harus berisi 60 data');
    }
    if (!data.future_7 || data.future_7.length !== 7) {
      throw new Error('future_7 harus berisi 7 data');
    }
    
    const payload = {
      version: v || 'version_0',
      encoder_last_60: data.encoder_last_60,
      future_7: data.future_7
    };
    
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const result = await resp.json();
    
    if (result.predictions) {
      window.lastPredictions = result.predictions;
      renderChart(result.predictions);
      renderTable(result.predictions);
      status.textContent = 'Prediksi selesai!';
      status.className = 'success';
    } else {
      status.textContent = 'Error: ' + (result.error || 'unknown');
      status.className = 'error';
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
    status.className = 'error';
  }
}

function renderChart(preds) {
  const xs = preds.map(p => 'Hari ' + preds.indexOf(p) + 1);
  const ys = preds.map(p => p.prediction);
  const trace = { 
    x: xs, 
    y: ys, 
    type: 'scatter', 
    mode: 'lines+markers', 
    name: 'Prediksi',
    line: { color: '#3498db', width: 3 },
    marker: { size: 10 }
  };
  const layout = { 
    title: 'Prediksi Harga Emas (7 Hari)',
    xaxis: { title: 'Hari' },
    yaxis: { title: 'Harga Prediksi' }
  };
  Plotly.newPlot('chart', [trace], layout);
}

function renderTable(preds) {
  const tbody = document.querySelector('#resultTable tbody');
  tbody.innerHTML = preds.map((p, i) => 
    `<tr><td>Hari ${i+1}</td><td>${p.time_idx}</td><td>${p.prediction.toFixed(2)}</td></tr>`
  ).join('');
}

window.onload = () => {
  loadVersions();
  document.getElementById('btnPredict').addEventListener('click', handlePredict);
};