# PLANNING.md — Rencana Pengerjaan Proyek MSML

## Fase 1: Setup & Dataset
- [ ] Setup Python virtual environment (Python 3.12.7)
- [ ] Install dependencies (pandas, numpy, scikit-learn, mlflow==2.19.0, dagshub, etc.)
- [ ] Download Pima Indians Diabetes dataset ke `diabetes_raw/`
- [ ] Setup Git dengan remote GitHub + DagsHub

## Fase 2: Kriteria 1 — Eksperimen (Target: Advance = 4 pts)
### Basic (2 pts)
- [ ] Buat folder `preprocessing/`
- [ ] Buat notebook `Eksperimen_Farrel-Ghozy-Affifudin.ipynb`
  - [ ] Data loading dari `diabetes_raw/diabetes.csv`
  - [ ] EDA (statistik deskriptif, visualisasi distribusi, korelasi)
  - [ ] Preprocessing (handle missing values, scaling, split)
  - [ ] Simpan hasil ke `diabetes_preprocessing/`

### Skilled (3 pts)
- [ ] Buat `automate_Farrel-Ghozy-Affifudin.py`
  - [ ] Fungsi load_data()
  - [ ] Fungsi eda()
  - [ ] Fungsi preprocessing()
  - [ ] Fungsi main() — pipeline lengkap

### Advance (4 pts)
- [ ] Buat `.github/workflows/preprocessing.yml`
  - [ ] Trigger: push ke branch main, path: diabetes_raw/
  - [ ] Steps: setup python → install deps → run automate.py → commit hasil preprocessing
- [ ] Buat repo `Eksperimen_SML_Farrel-Ghozy-Affifudin` di GitHub
- [ ] Push semua file ke repo tersebut

## Fase 3: Kriteria 2 — Membangun Model (Target: Advance = 4 pts)
### Basic (2 pts)
- [ ] Buat `modelling.py` dengan MLflow autolog
  - [ ] Load data preprocessing
  - [ ] Train RandomForestClassifier
  - [ ] MLflow autolog, simpan di localhost:5000
  - [ ] Screenshot dashboard MLflow

### Skilled (3 pts)
- [ ] Buat `modelling_tuning.py` dengan manual logging
  - [ ] Hyperparameter tuning (GridSearchCV / RandomizedSearch)
  - [ ] Manual logging: params, metrics, model artifact
  - [ ] Screenshot dashboard dengan runs yang berbeda

### Advance (4 pts)
- [ ] Setup DagsHub remote tracking
- [ ] `modelling_tuning.py` — manual logging + 2 artifacts tambahan:
  - [ ] Confusion matrix plot
  - [ ] Feature importance plot
  - [ ] ROC curve plot
- [ ] Screenshot dashboard DagsHub
- [ ] Simpan link DagsHub di `DagsHub.txt`

## Fase 4: Kriteria 3 — Workflow CI (Target: Advance = 4 pts)
### Basic (2 pts)
- [ ] Buat folder `MLProject/`
  - [ ] `MLProject` file
  - [ ] `conda.yaml`
  - [ ] `modelling.py` yang sudah disesuaikan
  - [ ] `diabetes_preprocessing/`

### Skilled (3 pts)
- [ ] Buat `.github/workflows/ci.yml`
  - [ ] Trigger: push ke main
  - [ ] Setup Python + dependencies
  - [ ] Train model via `mlflow run`
  - [ ] Upload artifacts ke GitHub Actions

### Advance (4 pts)
- [ ] Tambah step `mlflow models build-docker`
- [ ] Push Docker image ke Docker Hub
- [ ] Buat repo `Workflow-CI` di GitHub
- [ ] Push semua file

## Fase 5: Kriteria 4 — Monitoring & Logging (Target: Advance = 4 pts)
### Basic (2 pts)
- [ ] Serving model via MLflow / Docker
- [ ] `2.prometheus.yml` — scrape MLflow metrics
- [ ] `3.prometheus_exporter.py` — custom Prometheus metrics
- [ ] 3 metrics Prometheus (e.g., prediction count, latency, model confidence)
- [ ] Grafana dashboard (same 3 metrics)
- [ ] Screenshot bukti serving

### Skilled (3 pts)
- [ ] 5 metrics di Grafana
- [ ] 1 alerting rule di Grafana

### Advance (4 pts)
- [ ] 10 metrics di Grafana
- [ ] 3 alerting rules di Grafana
- [ ] Screenshot rules + notifikasi

## Fase 6: Finalisasi Submission
- [ ] Susun folder `SMSML_Farrel-Ghozy-Affifudin`
- [ ] Buat `Eksperimen_SML_Farrel-Ghozy-Affifudin.txt` (isi link repo GitHub)
- [ ] Buat `Workflow-CI.txt` (isi link repo GitHub)
- [ ] Verifikasi semua struktur sesuai ketentuan
- [ ] Push final ke GitHub

## Dependencies Utama
- Python 3.12.7
- mlflow==2.19.0
- scikit-learn, pandas, numpy, matplotlib, seaborn
- dagshub
- prometheus-client
- fastapi, uvicorn (untuk serving)
