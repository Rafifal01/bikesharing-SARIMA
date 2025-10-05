# README.md

# Bike Sharing Demand Forecasting

Proyek ini memprediksi jumlah peminjaman sepeda menggunakan model hybrid Gradient Boosting dan SARIMA, dengan data historis peminjaman serta fitur cuaca dan kalender.

## Struktur Folder

- `app.py` : Aplikasi Streamlit untuk prediksi dan visualisasi.
- `caps3.ipynb` : Notebook eksplorasi, training, evaluasi, dan pembuatan model.
- `data_bike_sharing.csv` : Dataset utama.
- `best_hourly_model.pkl` : Model Gradient Boosting hasil training.
- `caps_env/` : Virtual environment (jangan edit langsung).
- `.vscode/` : Konfigurasi editor.

## Cara Menjalankan

1. **Aktifkan environment**  
   Jalankan di terminal:

   ```sh
   source caps_env/Scripts/activate
   ```

   atau di Windows:

   ```sh
   caps_env\Scripts\activate
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit**
   ```sh
   uv run streamlit run app.py
   ```

## Fitur

- Prediksi jumlah rental sepeda harian, mingguan, atau bulanan.
- Model hybrid: menggabungkan prediksi Gradient Boosting dan SARIMA dengan bobot dinamis.
- Visualisasi hasil prediksi dan bobot model.

## Dataset

Pastikan file `data_bike_sharing.csv` tersedia di folder utama.

## Notebook

Lakukan eksplorasi, training, dan evaluasi model di [caps3.ipynb](caps3.ipynb).

## Kontak

Untuk pertanyaan, silakan hubungi pemilik repo.
