# Fraud Detection MLOps Platform

This project is an end-to-end Machine Learning Operations (MLOps) platform designed for the IEEE-CIS Fraud Detection dataset. It features a scalable, memory-efficient MLflow pipeline, advanced modeling (LightGBM), cost-sensitive learning, a production-ready CI/CD pipeline, comprehensive monitoring with Prometheus and Grafana, and automated model explainability (SHAP).

## Project Structure

```text
├── setup.sh                 # Environment setup & dependency installation
├── requirements.txt         # Python dependencies
├── src/                     # Core Machine Learning & API logic
│   ├── api.py               # FastAPI inference service with Prometheus metrics
│   ├── drift_simulation.py  # Simulates time-based data drift & triggers CI/CD
│   ├── explainability.py    # Generates SHAP explainability plots
│   └── (other core files)   # Preprocessing, features, models logic
├── pipeline/                # Orchestration
│   └── fraud_pipeline.py    # MLflow Sequential Pipeline definition
├── docker/                  # Containerization
│   └── Dockerfile.api       # Dockerfile for the Inference API
├── cicd/                    # CI/CD Workflows
│   └── .github/workflows/ci_cd_pipeline.yml # GitHub Actions pipeline
├── monitoring/              # Observability stack
│   ├── prometheus.yml       # Prometheus configuration
│   ├── alert.rules          # Prometheus alert definitions
│   └── grafana_dashboards/  # JSON templates for Grafana dashboards
├── tests/                   # Automated tests
│   └── test_api.py          # Pytest unit tests for the API
└── data/                    # Raw dataset (train_transaction, etc.)
```

## Prerequisites
- Ubuntu/Linux Environment
- At least 4GB of RAM (A swap file is configured in the setup script)
- Python 3.10+
- Docker (for running Prometheus and Grafana)

---

## 1. Environment Setup

First, execute the setup script. This will configure a system swap file to prevent out-of-memory errors during heavy data processing, create the necessary artifact directories, and install your Python dependencies (including MLflow).

```bash
chmod +x setup.sh
./setup.sh
```

---

## 2. Running the MLflow Pipeline (Model Training)

The core machine learning pipeline sequentially handles data ingestion, validation, preprocessing, feature engineering, and model training (XGBoost & LightGBM) while strictly managing memory consumption.

1. **Run the training pipeline:**
   ```bash
   python3 pipeline/fraud_pipeline.py
   ```
   *Note: Artifacts and trained models are automatically saved to `fraud-artifacts/`.*

2. **Start the MLflow tracking UI** to monitor execution, parameters, and view saved artifacts:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
3. Visit **`http://localhost:5000`** in your browser to view the tracking dashboard.

---

## 3. Serving the Model (Inference API)

To spin up the FastAPI service that serves the trained LightGBM model predictions and exposes live metrics:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Prometheus Metrics Endpoint**: `http://localhost:8000/metrics`

---

## 4. Monitoring & Observability (Grafana & Prometheus)

The monitoring stack allows you to track API performance, transaction latency, and data drift in real-time.

1. **Start Prometheus** (using the configuration from your project):
   ```bash
   docker run -d -p 9090:9090 -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```
2. **Start Grafana**:
   ```bash
   docker run -d -p 3000:3000 grafana/grafana
   ```
3. **Configure the Dashboard**:
   - Open Grafana at **`http://localhost:3000`** (Login: `admin`/`admin`)
   - Go to **Data Sources**, add **Prometheus**, and set the URL to `http://172.17.0.1:9090`.
   - Go to **Dashboards -> Import**, and upload the `monitoring/grafana_dashboards/system_health.json` file.

### Simulating Data Drift
To prove the system detects distribution shifts, you can fire a script that generates mocked "drifted" traffic to the API:
```bash
python3 src/drift_simulation.py
```
This triggers an alert in Prometheus and fires a simulated webhook to GitHub Actions.

---

## 5. Model Explainability (SHAP)

To understand exactly why the model flags certain transactions as fraudulent, run the SHAP explainability script. 

```bash
python3 src/explainability.py
```
This script evaluates the model and generates visual plots (Global Feature Importance, Global Impact, and Local Transaction Explanations). The high-quality visual outputs are automatically saved to the `fraud-artifacts/plots/` directory.

---

## 6. Continuous Integration / Continuous Deployment (CI/CD)

This repository includes a fully configured CI/CD pipeline powered by **GitHub Actions**. The configuration is located at `cicd/.github/workflows/ci_cd_pipeline.yml`.

### How the CI/CD Pipeline works:
1. **Automated Testing & Linting**: Whenever you push code to the `main` branch, GitHub Actions will automatically spin up an environment, install dependencies, and run your tests (`pytest tests/test_api.py`) to ensure the FastAPI endpoints are healthy.
2. **Automated Retraining via Drift Webhooks**: The pipeline is configured to accept `repository_dispatch` events. When your `drift_simulation.py` script detects significant data drift, it fires a webhook directly to GitHub Actions. This tells GitHub to automatically start a job that rebuilds your environment and retrains your model on the new data distribution.

### How to run the CI/CD Pipeline:
1. **Push your code to GitHub**: 
   Ensure this project is initialized as a Git repository and pushed to GitHub. The action will trigger automatically on your first push.
   ```bash
   git add .
   git commit -m "Initial MLOps commit"
   git push origin main
   ```
2. **View the Action**: 
   Navigate to the **"Actions"** tab on your GitHub repository page to watch the pipeline execute in real-time.
3. **Configure Webhook Secrets (For Drift Retraining)**:
   To allow the drift script to trigger retraining, you must generate a GitHub Personal Access Token (PAT), add it as a repository secret named `PAT_TOKEN`, and update `src/drift_simulation.py` to point to your specific GitHub repository URL.
