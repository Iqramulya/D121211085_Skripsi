# TFT Bersih

Salinan runtime website Flask prediksi harga emas dengan model TFT training ke-5 / versi 2 (ID8 run3). Tidak memerlukan folder proyek asal untuk kode, checkpoint, dataset, maupun hasil evaluasi.

## Isi folder

- `app.py`: entry point Flask; port bawaan `5001`, dapat diubah melalui `PORT`.
- `app/`: kode backend, preprocessing, inferensi, skenario, dan evaluasi.
- `templates/`: antarmuka website terbaru.
- `static/`: aset website.
- `models/training_v2/version_0/`: checkpoint aktif dan metadata asal model.
- `models/training_v2/gold_price_no_idr_update.csv`: dataset bawaan untuk Load Dataset dan baseline input manual.
- `models/training_v2/forecast_actuals.csv`: harga aktual pembanding tersimpan dari notebook.
- `models/training_v2/hasil_evaluasi_tft/`: metrik dan grafik evaluasi training.
- `gold_price_no_idr.csv`: sumber harga aktual dan tampilan riwayat sesuai perilaku aplikasi asal.
- `gold_price_no_idr_update.csv`: dataset untuk unggahan manual dan sumber riwayat cadangan.
- `requirements.txt`, `requirements_cpu.txt`: daftar dependensi dari proyek asal.

Model lama, arsip ZIP training, notebook training, log training, eksperimen, backup, cache, virtual environment, serta konfigurasi pribadi tidak disertakan. Nama arsip dan notebook pada `source.json` adalah catatan asal model, bukan berkas yang diperlukan untuk menjalankan website.

## Menjalankan

Dari dalam folder `tft_bersih`, buat lingkungan Python baru. Verifikasi salinan ini memakai Python 3.13 dan dependensi yang sudah terpasang pada lingkungan proyek asal; instalasi ulang dependensi belum diuji.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Buka `http://localhost:5001`. Jika port tersebut sudah dipakai:

```sh
PORT=5002 .venv/bin/python app.py
```

`app.py` menjalankan development server dengan debug aktif; jangan gunakan langsung untuk deployment publik. Contoh menjalankan dengan Gunicorn setelah dependensi terpasang:

```sh
.venv/bin/gunicorn --bind 127.0.0.1:5002 'app:create_app()'
```

Grafik interaktif dan font menggunakan CDN sehingga memerlukan internet. Endpoint data pasar langsung juga memerlukan akses layanan eksternal.

## Data dan tanggal prediksi

Dataset bawaan berisi 4.368 observasi, dari 1 Januari 2010 hingga 12 Oktober 2026. Prediksi berlanjut setelah tanggal terakhir input selama **7 Hari Kerja** (Senin–Jumat); Sabtu–Minggu dilewati, tanpa kalender libur bursa. Dengan dataset bawaan, tanggal prediksi adalah 13, 14, 15, 16, 19, 20, dan 21 Oktober 2026.
Antarmuka memakai label **7 Hari**; label singkat ini tidak mengubah perhitungan tanggal prediksi di atas.

Gunakan Upload CSV untuk memprediksi dari riwayat terbaru tanpa melatih ulang model. Unggah riwayat lengkap dengan format kolom yang sesuai, bukan hanya baris baru. Pipeline membutuhkan minimal 80 observasi mentah untuk membentuk fitur dan encoder 60 observasi.

Tanggal tabel menyertakan tahun. Paragraf penjelasan rumus error di bawah tabel sudah dihapus; kolom error tetap tersedia.

## Verifikasi salinan

Salinan dijalankan dari folder ini memakai lingkungan Python proyek asal, tanpa menyalin `.venv` dan tanpa override lokasi model/dataset:

- Load Dataset dan prediksi melalui browser: tujuh hasil TFT, 13–21 Oktober 2026; prediksi pertama USD 4.109,8628.
- Endpoint `/health`, `/model-info`, `/sample-scenarios`, `/historical-performance`, `/evaluation-notebook-chart`, dan aset `/static/evaluation_reference.png`: HTTP 200.
- `POST /evaluate`: tujuh horizon, MAE 68,5037.

Akses data pasar langsung dari layanan eksternal tidak termasuk verifikasi ini.
