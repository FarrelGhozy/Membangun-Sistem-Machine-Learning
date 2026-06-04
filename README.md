# Sistem Machine Learning - Farrel Ghozy Affifudin

Proyek submission kelas "Membangun Sistem Machine Learning" Dicoding.

## Struktur Proyek

```
SMSML_Farrel-Ghozy-Affifudin/
├── Eksperimen_SML_Farrel-Ghozy-Affifudin.txt  (link repo GitHub Kriteria 1)
├── Membangun_model/                            (Kriteria 2)
│   ├── modelling.py                            (Basic: autolog MLflow)
│   ├── modelling_tuning.py                     (Skilled: tuning + manual log)
│   ├── diabetes_preprocessing/                 (data siap latih)
│   ├── screenshoot_dashboard.jpg               (screenshot MLflow UI)
│   ├── screenshoot_artifak.jpg                 (screenshot artifacts)
│   ├── requirements.txt
│   └── DagsHub.txt                             (link DagsHub)
├── Workflow-CI.txt                             (link repo GitHub Kriteria 3)
└── Monitoring dan Logging/                     (Kriteria 4)
    ├── serving.py                              (FastAPI serving)
    ├── 1.bukti_serving
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    ├── 6.bukti alerting Grafana/
    └── 7.Inference.py
```

## Cara Menjalankan

### 1. Preprocessing
```bash
source venv/bin/activate
python preprocessing/automate_Farrel-Ghozy-Affifudin.py \
  --input diabetes_raw/diabetes_with_headers.csv \
  --output preprocessing/diabetes_preprocessing
```

### 2. Training Model
```bash
# Basic (autolog)
cd Membangun_model && python modelling.py

# Tuning (manual log + hyperparameter tuning)
cd Membangun_model && python modelling_tuning.py

# Tuning dengan DagsHub (Advance)
cd Membangun_model && python modelling_tuning.py --dagshub
```

### 3. MLflow UI
```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri file:./Membangun_model/mlruns
```
Buka http://localhost:5000

### 4. Serving & Monitoring
```bash
# Terminal 1: Start API
python "Monitoring dan Logging/serving.py"

# Terminal 2: Test inference
python "Monitoring dan Logging/7.Inference.py"

# Terminal 3: Prometheus exporter
python "Monitoring dan Logging/3.prometheus_exporter.py"
```

### 5. Prometheus & Grafana
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Akun
- **Nama**: Farrel Ghozy Affifudin
- **Username Dicoding**: fazyfif
- **Dashboard Grafana**: fazyfif
