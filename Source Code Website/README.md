# Gold Price Predictor — salinan runtime bersih

Folder ini berisi kode dan artefak yang dibutuhkan untuk menjalankan website
dengan model TFT training ke-5 / versi 2 (ID8 run3). Folder dapat dipindahkan
secara utuh tanpa bergantung pada proyek asal. Arsip training, notebook, backup,
model lama, log, cache, dan virtual environment tidak disertakan.

## Instalasi dan menjalankan

Gunakan Python 3.11–3.13. Dari folder ini:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m flask --app app.py run --host 127.0.0.1 --port 5002
```

Buka `http://127.0.0.1:5002`. Untuk deployment gunakan WSGI server, bukan server
development Flask. Internet diperlukan untuk Yahoo Finance, Plotly CDN,
dan Google Fonts.

## Berkas penting

```text
app.py
app/                         # Backend, preprocessing, inferensi, aktual, evaluasi
templates/index.html         # Antarmuka website
models/training_v2/
  version_0/
    source.json              # Metadata dan SHA-256 checkpoint aktif
    checkpoints/
      epoch=2-step=324.ckpt
  hasil_evaluasi_tft/         # CSV metrik dan grafik evaluasi
  gold_price_no_idr_update.csv -> ../../gold_price_no_idr_update.csv
gold_price_no_idr_update.csv  # Satu dataset historis yang disalin
aktual.csv -> gold_price_no_idr_update.csv
tests/                       # Regresi aktual dan antarmuka/input
requirements.txt
README.md
```

Kedua symlink relatif menunjuk ke berkas dalam folder ini, bukan ke proyek asal.
Pertahankan symlink ketika memindahkan folder, atau salin isi target jika
lingkungan tujuan tidak mendukung symlink.

## Model dan data historis

Checkpoint aktif menyimpan scaler, encoder kategori, normalizer, dan parameter
dataset; tidak memerlukan berkas preprocessing eksternal. `source.json` mencatat
asal artefak training dan hasil forecast referensi. Arsip/notebook yang disebut
di metadata merupakan provenance, bukan dependensi runtime.

Dataset berisi 4.368 observasi pada 2010-01-01 hingga 2026-10-12. Preprocessing
menghasilkan 4.348 baris sebelum penambahan horizon. Forecast menggunakan encoder
60 observasi dan tujuh hari kerja sesudah tanggal terakhir data input.
Dengan dataset bawaan, hasil pertama adalah 4109,86 USD untuk 13 Oktober 2026.
`TFT_SCENARIO_CSV` dapat menentukan CSV historis lain; `TFT_MODEL_BASE` atau
`TFT_LIGHTNING_LOGS` dapat menentukan direktori model lain.

## Memilih periode prediksi

1. Muat data melalui Upload CSV, Load Dataset, Manual Input, atau Sample Data.
2. Pilih **Tanggal akhir data historis** sebelum Generate Forecast. Batas minimum
   menyisakan 80 observasi mentah dan batas maksimum mengikuti sumber.
3. Periksa pratinjau tanggal riwayat efektif, periode tujuh hari kerja forecast,
   dan sumber/ketersediaan aktual, lalu klik Generate Forecast.

Contoh: batas **30 September 2026** menghasilkan forecast **1–9 Oktober 2026**.
Data setelah batas tidak masuk TFT. Tanggal tanpa observasi menggunakan observasi
terakhir pada/sebelum pilihan, tanpa menambah data buatan.

Setiap mode mengingat pilihannya selama halaman terbuka; pemuatan ulang sumber
mengembalikan pilihan ke tanggal terakhir. Mengubah tanggal pada mode manual
mengisi ulang indikator dari observasi terpilih sebelum pengguna mengubah nilainya.
Mengganti tanggal mengosongkan tampilan dan mengabaikan respons prediksi lama;
forecast tersimpan baru diganti setelah prediksi baru berhasil.

`GET /api/actuals-coverage` menjelaskan cakupan CSV saat halaman dimuat dan sebelum
Generate, tanpa model/Yahoo. Pemilih tanggal tidak mengisi aktual yang belum ada.

## Harga aktual dan error

- **13–21 Oktober 2026, inklusif:** hanya `aktual.csv` dengan kolom `date,close`,
  tanggal `YYYY-MM-DD`, dan harga positif.
- **Tanggal lain:** daily `Close` Yahoo Finance `GC=F`, dicocokkan tepat tanggalnya.
  Bar hari berjalan baru dipakai mulai pukul 17:00 America/New_York sebagai batas
  konservatif selesainya sesi. Harga intraday sebelumnya tidak dianggap penutupan.
- **Aktual tidak tersedia:** prediksi tetap tampil; aktual, error USD, dan error
  persen menjadi `-`. Tidak ada penggantian harga dengan tanggal lain.
- **Rumus:** error USD = `abs(prediksi - aktual)`; error persen =
  `abs(prediksi - aktual) / aktual * 100`.

Sumber sementara `aktual.csv` masih menunjuk dataset yang berakhir pada
**12 Oktober 2026**. Karena tidak ada baris 13–21 Oktober, aktual/error untuk
forecast bawaan tetap `-`. Saat file aktual asli tersedia, ganti symlink dengan
file tersebut; jangan menulis melalui symlink karena akan menimpa dataset.

Pembaruan aktual dilakukan setiap 60 detik selama halaman terlihat, saat kembali
terlihat, dan setelah pemulihan forecast dari `localStorage` (`tft:last-forecast`).
Pembaruan tidak menjalankan ulang TFT atau mengganti prediksi tersimpan.
Forecast tersimpan berlaku pada browser/origin yang sama, bukan riwayat server.

## Evaluasi dan pengujian

Evaluasi adalah hasil validasi tersimpan dari training, bukan evaluasi ulang CSV
unggahan: MAE 68,5037; RMSE 107,6815; MAPE 1,97%; R² 0,9884; 6.258 pasangan.

```sh
.venv/bin/python -m pytest tests/test_forecast_actuals.py -q
node --test tests/test_manual_input.cjs
```

Node.js diperlukan hanya untuk menjalankan tes JavaScript. Salinan ini memuat
regresi aktual dan antarmuka/input, bukan seluruh suite penelitian proyek asal.

Endpoint utama: `/`, `/health`, `/versions`, `/status`, `/load-dataset`,
`/sample-scenarios`, `/model-info`, `/historical-performance`, `POST /predict`,
`GET /api/actuals-coverage`, `POST /api/forecast-actuals`, `POST /evaluate`,
`/evaluation-notebook-chart`.
