# Beginner Guide: Task 5 to Task 9

This guide is written for someone starting from zero. You can follow it line by line.

## What You Will Build

By the end, you will have:
1. A CI CD pipeline that runs tests, builds images, and triggers Kubeflow retraining.
2. Monitoring with Prometheus and dashboards in Grafana.
3. Alert rules that detect model and data problems.
4. Automatic retraining triggers when alerts fire.
5. Drift simulation to prove your monitoring is working.
6. SHAP explainability outputs for model decisions.

---

## Task 5: CI CD Pipeline with Intelligent Triggers

## 1. CI CD Basics in One Minute

CI means Continuous Integration.
1. Every time you push code or open a pull request, checks run automatically.
2. Checks include linting, tests, and data validation.

CD means Continuous Deployment.
1. After checks pass, your system builds and deploys updates.
2. In this project, CD triggers Kubeflow retraining and API rollout.

## 2. Put Workflow in Correct GitHub Folder

GitHub Actions only auto-runs files in:
1. .github/workflows at repository root

Your workflow currently exists at:
1. [cicd/.github/workflows/ci_cd_pipeline.yml](cicd/.github/workflows/ci_cd_pipeline.yml)

Create this root folder and copy the workflow there:
1. .github/workflows/ci_cd_pipeline.yml

## 3. Understand Your Workflow Stages

Your workflow should do the following:

Stage 1 CI:
1. Trigger on push and pull request.
2. Run lint and unit tests.
3. Run data validation checks.

Stage 2 Build and Packaging:
1. Build inference API Docker image from [docker/Dockerfile.api](docker/Dockerfile.api).
2. Build training Docker image from [docker/Dockerfile.training](docker/Dockerfile.training).
3. Push both images to registry.

Stage 3 Continuous Deployment:
1. Compile and submit Kubeflow pipeline from [pipeline/fraud_pipeline.py](pipeline/fraud_pipeline.py).
2. Use KFP host and token from GitHub Secrets.

Stage 4 Intelligent Trigger:
1. Accept repository_dispatch events.
2. Retrain automatically when alert bridge sends model_performance_drop or data_drift_alert.

## 4. Configure Required GitHub Secrets

In GitHub repo settings, add:
1. KFP_HOST
2. KFP_TOKEN
3. KUBECONFIG_B64
4. K8S_NAMESPACE
5. K8S_DEPLOYMENT_NAME
6. K8S_CONTAINER_NAME

Optional note:
1. KFP_TOKEN can be omitted only if your Kubeflow endpoint allows access without token.

## 5. Validate CI CD End to End

Do this sequence:
1. Push a small commit to main.
2. Open GitHub Actions tab.
3. Confirm Stage 1 passes.
4. Confirm Stage 2 pushes images.
5. Confirm Stage 3 submits Kubeflow run.
6. Confirm Stage 4 updates deployment image and rollout succeeds.

## 6. Collect Evidence for Task 5

Save screenshots or logs for:
1. Push trigger run.
2. Pull request trigger run.
3. Repository dispatch run.
4. Kubeflow run ID created from CI.
5. Deployment rollout success.

---

## Task 6: Observability and Monitoring with Prometheus and Grafana

## 1. Monitoring Basics in One Minute

Prometheus:
1. Collects metrics from services.
2. Evaluates alert rules.

Grafana:
1. Reads Prometheus data.
2. Visualizes dashboards and alerts.

## 2. Use Existing Project Files

You already have:
1. [monitoring/prometheus.yml](monitoring/prometheus.yml)
2. [monitoring/alert.rules](monitoring/alert.rules)
3. [monitoring/grafana_dashboards/system_health.json](monitoring/grafana_dashboards/system_health.json)
4. [src/api.py](src/api.py)

## 3. Verify API Exposes Metrics

In [src/api.py](src/api.py), verify:
1. /metrics endpoint exists.
2. Request counters and latency histograms are recorded.
3. Prediction counters and confidence metrics are recorded.
4. Data drift gauge exists.

## 4. Start Prometheus and Verify Scrape

Basic checks:
1. Start your API service.
2. Open API /metrics endpoint in browser.
3. Open Prometheus UI.
4. In Prometheus targets page, verify fraud_inference_api target is UP.

## 5. Import Grafana Dashboard

In Grafana:
1. Add Prometheus as data source.
2. Import [monitoring/grafana_dashboards/system_health.json](monitoring/grafana_dashboards/system_health.json).
3. Confirm charts show latency, throughput, and resource health.

## 6. Build Three Required Dashboards

System Health dashboard:
1. API request rate
2. Latency
3. Error rate
4. CPU and memory

Model Performance dashboard:
1. Recall trend over time
2. Fraud detection rate
3. Precision recall trade-off
4. Confidence distribution

Data Drift dashboard:
1. Drift score trend
2. Missing value trend
3. Input anomaly indicator

## 7. Configure and Validate Alert Rules

Use [monitoring/alert.rules](monitoring/alert.rules) and ensure alerts exist for:
1. Recall drop below threshold
2. Data drift above threshold
3. Latency spike

Then test:
1. Generate traffic spike for latency alert.
2. Trigger drift simulation for drift alert.
3. Confirm alert appears in Prometheus and Grafana.

## 8. Collect Evidence for Task 6

Save:
1. Prometheus config screenshot.
2. Alert rules screenshot.
3. Three dashboard screenshots.
4. Alert firing screenshot.
5. Logs showing alert action triggered.

---

## Task 7: Realistic Drift Simulation

## 1. What This Means

Instead of random noise, you simulate real drift by time:
1. Train on earlier data distribution.
2. Test on later data distribution.
3. Add new fraud behavior patterns in later window.

## 2. Use Existing Drift Script

Run:
1. [src/drift_simulation.py](src/drift_simulation.py)

This script already:
1. Splits early and late data.
2. Alters fraud-related feature patterns in late data.
3. Writes drifted datasets.
4. Signals drift alert metric.

## 3. Output Files to Verify

Check generated files in artifacts folder:
1. drift_sim_train.csv
2. drift_sim_test.csv

## 4. What to Capture

1. Console output showing drift simulation completed.
2. Evidence of changed distributions.
3. Drift alert firing evidence.

---

## Task 8: Intelligent Retraining Strategy

## 1. Strategy Types

Threshold-based:
1. Retrain when recall or drift crosses threshold.

Periodic:
1. Retrain on fixed schedule.

Hybrid:
1. Periodic retrain plus emergency retrain on alerts.

## 2. Recommended for Your Project

Use Hybrid because it balances stability and responsiveness.

## 3. Practical Implementation Steps

1. Keep scheduled trigger in workflow for periodic retraining.
2. Keep repository_dispatch triggers for emergency retraining.
3. Add cooldown logic to avoid repeated retraining loops.
4. Define acceptance criteria for new model before deployment.

Example acceptance criteria:
1. recall does not drop below baseline threshold
2. business cost improves or stays acceptable
3. no severe precision collapse

## 4. Update Strategy Report

Document in [report/retraining_strategy.md](report/retraining_strategy.md):
1. Trigger conditions
2. Cooldown window
3. Rollback conditions
4. Cost and performance comparison

## 5. Evidence for Task 8

1. One periodic retraining run log.
2. One emergency alert-driven retraining run log.
3. Before versus after metrics comparison table.

---

## Task 9: Explainability Requirement

## 1. Goal

Answer: Why did the model predict fraud?

## 2. Use Existing Explainability Script

Run:
1. [src/explainability.py](src/explainability.py)

It should produce:
1. Global feature importance plot
2. Global impact direction plot
3. Local explanation for one high-risk transaction

## 3. Important Check Before Running

Ensure explainability script loads the same model filename that training actually saves.

If filenames differ, update [src/explainability.py](src/explainability.py) model path so it points to your current best model artifact.

## 4. What to Write in Report

Global explanation:
1. Top features that drive fraud predictions.

Local explanation:
1. For one transaction, identify features pushing prediction toward fraud.

Business interpretation:
1. Explain how those features align with fraud behavior logic.

## 5. Evidence for Task 9

1. SHAP plot images from artifacts.
2. One short interpretation paragraph per plot type.

---

## Final Demo Flow You Can Present

Use this order in your final presentation:
1. Show CI run on push.
2. Show image build and push.
3. Show Kubeflow run auto-trigger.
4. Show Prometheus and Grafana dashboards live.
5. Trigger drift simulation and show alert firing.
6. Show repository_dispatch retraining run starts automatically.
7. Show post-retraining metrics.
8. Show SHAP explainability outputs.

---

## Quick Troubleshooting

Workflow not running:
1. Confirm file is in root .github/workflows.
2. Confirm branch filter includes your branch.

Kubeflow trigger failing:
1. Check KFP_HOST and KFP_TOKEN secrets.
2. Verify network access from GitHub runner to Kubeflow.

No metrics in Grafana:
1. Check Prometheus target health.
2. Check API /metrics endpoint returns data.

Alert not triggering CI:
1. Check alert expression and threshold.
2. Check Alertmanager webhook receiver logs.
3. Check GitHub token permissions for repository_dispatch.

Explainability script fails:
1. Confirm SHAP is installed.
2. Confirm model path and feature columns match.

---

## Submission Checklist

Task 5:
1. CI CD workflow YAML
2. Automated run logs
3. Evidence of automated retraining

Task 6:
1. Prometheus config
2. Alert rules
3. Grafana screenshots
4. Alert trigger evidence

Task 7:
1. Drift simulation outputs
2. Drift alert evidence

Task 8:
1. Retraining strategy document
2. Stability cost performance comparison

Task 9:
1. SHAP plots
2. Explainability write-up
