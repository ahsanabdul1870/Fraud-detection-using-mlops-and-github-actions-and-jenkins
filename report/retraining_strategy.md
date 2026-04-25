# Intelligent Retraining Strategy

In an MLOps environment, model decay (concept drift and data drift) is inevitable. For the IEEE-CIS Fraud Detection system, maintaining high recall over time requires a robust retraining strategy. This document compares three primary approaches: Threshold-based, Periodic, and Hybrid.

## 1. Threshold-Based Retraining
This strategy triggers a new Kubeflow pipeline run only when specific metrics cross predefined thresholds.

- **Triggers**:
  - **Performance Drop**: E.g., Fraud Recall drops below 80% or F1-score drops below 75% on a recent validation batch.
  - **Data Drift**: E.g., The distribution of transaction amounts (`TransactionAmt`) shifts by more than a set threshold (measured via KL divergence or KS statistic), triggering the `DataDriftDetected` Prometheus alert.
- **Pros**:
  - Cost-efficient: Compute resources are only used when absolutely necessary.
  - Responsive: Can catch sudden fraud pattern changes immediately.
- **Cons**:
  - Stability risk: If the threshold is too sensitive, it can cause constant "flapping" (continuous retraining loops).
  - Delayed labels: Relies heavily on ground-truth labels arriving quickly, which is often difficult in fraud detection.

## 2. Periodic Retraining
This strategy retrains the model on a fixed schedule (e.g., daily, weekly, monthly).

- **Triggers**: Time-based cron job.
- **Pros**:
  - Highly stable and predictable.
  - Easy to implement and schedule CI/CD resources.
  - Does not rely on complex drift detection monitoring.
- **Cons**:
  - Inefficient: May retrain models that are still performing perfectly well, wasting compute cost and time.
  - Slow response: If a new fraud pattern emerges right after a retraining cycle, the system will be vulnerable until the next cycle.

## 3. Hybrid Retraining Strategy (Recommended)
The hybrid approach combines both strategies to balance cost, stability, and responsiveness.

- **Design**:
  - **Base Schedule**: Retrain periodically (e.g., bi-weekly) to capture slow, gradual drift (covariate shift) and incorporate new legitimate transaction patterns.
  - **Emergency Override**: Use threshold-based triggers linked to Prometheus alerts for sudden performance drops or significant data drift (e.g., the introduction of a new fraud ring). If an emergency trigger fires, it resets the periodic timer.
- **Pros**:
  - Best performance improvement: Captures both slow decay and sudden shocks.
  - Balances stability with responsiveness.
- **Cons**:
  - More complex to implement and maintain.
  - Requires tuning both the schedule and the thresholds.

## Conclusion and Implementation
For this fraud detection platform, the **Hybrid Strategy** is optimal.
1. The GitHub Actions workflow (`ci_cd_pipeline.yml`) is configured to accept `repository_dispatch` events triggered by Prometheus Alerts (Task 6) for Emergency overrides.
2. A scheduled cron trigger (`schedule: - cron: '0 0 * * 0'`) can be easily added to the same GitHub action for the periodic base schedule.
3. This ensures the business remains protected against sudden new fraud vectors without manually monitoring the pipeline 24/7.

## Task 4: Cost-Sensitive Learning Results and Business Impact

### Objective
Task 4 requires assigning a higher penalty to false negatives and comparing standard training against cost-sensitive training.

### Experimental Results

| Model | Precision | Recall | F1 | AUC-ROC | Cost | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (Standard) | 0.2616 | 0.8374 | 0.3987 | 0.9480 | 164,890 | 3,461 | 9,769 | 104,206 | 672 |
| XGBoost (Cost-Sensitive) | 0.1755 | 0.8921 | 0.2932 | 0.9477 | 217,870 | 3,687 | 17,327 | 96,648 | 446 |
| LightGBM (Standard) | 0.2782 | 0.8565 | 0.4200 | 0.9582 | 151,160 | 3,540 | 9,186 | 104,789 | 593 |
| LightGBM (Cost-Sensitive) | 0.1875 | 0.9049 | 0.3106 | 0.9577 | 201,370 | 3,740 | 16,207 | 97,768 | 393 |
| Hybrid (RF→LightGBM) | 0.2118 | 0.8396 | 0.3383 | 0.9411 | 195,420 | 3,470 | 12,912 | 101,063 | 663 |

Business cost definition used in this experiment:

Cost = FN x 100 + FP x 10

### Analysis: Standard vs Cost-Sensitive

1. Cost-sensitive training improved fraud recall for both model families.
2. XGBoost recall increased from 0.8374 to 0.8921, while FN dropped from 672 to 446.
3. LightGBM recall increased from 0.8565 to 0.9049, while FN dropped from 593 to 393.
4. This improvement came with a large increase in false positives.
5. XGBoost FP increased by 7,558 and LightGBM FP increased by 7,021.
6. Under the current cost weights, total business cost increased for both cost-sensitive variants.

### Business Impact Summary

- If business priority is maximum fraud capture (higher recall, fewer missed frauds), cost-sensitive variants are preferable.
- If business priority is minimum total operational cost under the current cost function, standard LightGBM is the best option.
- The best overall model under current assumptions is LightGBM (Standard), with the lowest cost (151,160) and the highest AUC-ROC (0.9582).

### Final Recommendation for Deployment

Deploy **LightGBM (Standard)** as the production default under current business weights.

Keep **LightGBM (Cost-Sensitive)** as an emergency policy option when fraud-loss risk increases and the organization is willing to accept more false alarms to reduce missed fraud cases.
