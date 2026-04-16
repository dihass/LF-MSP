# Leakage-Free Multimodal Sepsis Prediction (LF-MSP)

## Overview

This project implements a leakage-free multimodal sepsis early warning system that integrates structured Electronic Health Record (EHR) time-series data with clinical text. The system combines a two-layer LSTM, Bio_ClinicalBERT embeddings, and a logistic regression meta-model to produce early sepsis risk predictions with unified explainability.

The model predicts sepsis up to six hours prior to onset while enforcing strict temporal constraints to prevent data leakage, a common issue in retrospective clinical machine learning systems .

---

## Objectives

* Develop a leakage-free multimodal prediction pipeline
* Integrate structured EHR and unstructured clinical notes
* Provide unified, clinician-centered explanations
* Evaluate performance under strict temporal validation

---

## System Architecture

The system consists of four primary components:

### 1. Temporal Preprocessing

* Windowed EHR sequences (18-hour observation window)
* Strict masking of post-onset data
* Timestamp-based filtering of clinical notes

### 2. Modality Encoding

* LSTM for structured time-series data
* Bio_ClinicalBERT for clinical text representation

### 3. Fusion Layer

* Logistic regression meta-model combining modality outputs
* Robust handling of missing modalities

### 4. Explainability

* Gradient-based attribution for LSTM inputs
* TF-IDF-based term importance for text
* Fusion-level contribution analysis

---

## Performance

Evaluation on a held-out test set shows:

* AUROC: 0.9770
* AUPRC: 0.6071
* Sensitivity: 0.8047
* Negative Predictive Value (NPV): 0.9945

These results were obtained under strict leakage control and demonstrate strong predictive capability relative to baseline methods .

---

## Technology Stack

* Backend: FastAPI, Uvicorn
* Machine Learning: PyTorch, scikit-learn
* NLP: Hugging Face Transformers (Bio_ClinicalBERT)
* Data Processing: pandas, NumPy
* Visualization: matplotlib
* Deployment: Docker

---

## Project Structure

```
sepsis_app/
├── model_artifacts/      # Trained models and preprocessing artifacts
├── static/               # Frontend assets
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── setup.sh              # Environment setup script
├── run                   # Application run script
```

---

## Local Setup

```bash
./setup.sh
source venv/bin/activate
./run
```

---

## Docker Deployment

```bash
docker build -t sepsis-app .
docker run -p 8000:8000 sepsis-app
```

---

## API

* Swagger UI: http://localhost:8000/docs
* Endpoint: POST /predict

The API accepts ICU patient data in CSV format and returns:

* Sepsis risk probability
* Model explanations
* Supporting visualizations

---

## Dataset

This project is based on the MIMIC-IV v3.1 dataset, accessed via PhysioNet under appropriate data use agreements.

---

## Notebooks

Full pipeline and experiments are available via Google Colab:

Data Extraction and Observation Window Sensitivity
https://colab.research.google.com/drive/19ZVd51yaJM-8iih4zjh_e_nqjWK246He?usp=sharing
Model Training (LSTM and Meta-Fusion)
https://colab.research.google.com/drive/1rtiGnGp7_e-l7BT6FYaMwxFye780y5kG?usp=sharing

---

## Limitations

* Evaluation limited to a single dataset (MIMIC-IV)
* No external validation across institutions
* Transformer-based inference introduces latency

---

## Future Work

* External validation on multi-site datasets
* Real-time clinical integration
* Model optimization for lower latency
* Advanced multimodal fusion techniques

---

## Author

Dihas Sathnindu
BSc Computer Science
Informatics Institute of Technology (IIT)
University of Westminster

---

## License

This project is intended for academic and research purposes.
