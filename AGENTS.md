# AGENTS.md — Panduan Agent AI untuk Proyek MSML

## Identitas Proyek
- **Nama**: Farrel Ghozy Affifudin
- **Username Dicoding**: fazyfif
- **Email**: farrelafif05@gmail.com
- **Dataset**: Pima Indians Diabetes (Kaggle/UCI)
- **Target Nilai**: ADVANCE (4 pts) semua kriteria

## Struktur Folder Final Submission
```
SMSML_Farrel-Ghozy-Affifudin/
├── Eksperimen_SML_Farrel-Ghozy-Affifudin.txt (link repo GitHub Kriteria 1)
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── diabetes_preprocessing/
│   ├── screenshoot_dashboard.jpg
│   ├── screenshoot_artifak.jpg
│   ├── requirements.txt
│   └── DagsHub.txt (link DagsHub)
├── Workflow-CI.txt (link repo GitHub Kriteria 3)
└── Monitoring dan Logging/
    ├── 1.bukti_serving
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    ├── 6.bukti alerting Grafana/
    └── 7.Inference.py
```

## Aturan Kerja
1. Semua file dikerjakan dari direktori ini (`/home/fazy/Documents/DICODING/Pijak/SistemML`)
2. Dataset Pima Indians Diabetes akan didownload dari Kaggle
3. Git remote: GitHub (origin) + DagsHub (dagshub)
4. Setiap kriteria dikerjakan berurutan (1 → 2 → 3 → 4)
5. Screenshot diambil di akhir setiap tahap
6. Semua kode production-grade (type hints, error handling, logging)
