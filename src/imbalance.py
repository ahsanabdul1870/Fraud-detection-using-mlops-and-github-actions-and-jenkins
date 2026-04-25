"""
imbalance.py — Task 2: Class Imbalance Handling

Compares three strategies required by the assignment:
  A. SMOTE oversampling
  B. Random undersampling
  C. XGBoost class-weighting (scale_pos_weight)

Reads from train_features.csv (post feature-engineering) so the pipeline
is consistent with models.py.

NOTE: SMOTE on 590 K rows is very RAM-intensive. A 20% stratified sample
is used by default for SMOTE only (configurable via SMOTE_SAMPLE_FRAC).
Class-weighting and undersampling operate on the full training split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, recall_score,
    precision_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
import gc

ARTIFACT_DIR = os.environ.get(
    "OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts")
)

# Use a sample for SMOTE to avoid OOM on large dataset
SMOTE_SAMPLE_FRAC = float(os.environ.get("SMOTE_SAMPLE_FRAC", "0.2"))


# ── Helpers ───────────────────────────────────────────
def reduce_mem(df):
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ── Load data ─────────────────────────────────────────
def load_data(artifact_dir: str = ARTIFACT_DIR):
    """
    BUG FIX: Read from train_features.csv (output of features.py),
    NOT train_preprocessed.csv, so imbalance experiments use the same
    feature set as the final models.
    Falls back to train_preprocessed.csv if features file is missing.
    """
    features_path    = f"{artifact_dir}/train_features.csv"
    preprocessed_path = f"{artifact_dir}/train_preprocessed.csv"

    if os.path.exists(features_path):
        print(f"Loading from: {features_path}")
        df = reduce_mem(pd.read_csv(features_path))
    elif os.path.exists(preprocessed_path):
        print(f"WARNING: train_features.csv not found. "
              f"Falling back to {preprocessed_path}. "
              f"Run features.py first for best results.")
        df = reduce_mem(pd.read_csv(preprocessed_path))
        # Drop non-feature cols that features.py would have removed
        drop_cols = [c for c in ["TransactionID", "TransactionDT"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    else:
        raise FileNotFoundError(
            "Neither train_features.csv nor train_preprocessed.csv found. "
            "Run preprocessing.py then features.py first."
        )

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    fraud_rate = y.mean() * 100
    print(f"  Loaded: X={X.shape}, Fraud rate: {fraud_rate:.2f}% "
          f"({y.sum()} fraud / {len(y)} total)")
    del df
    gc.collect()
    return X, y


# ── Train + Evaluate helper ───────────────────────────
def train_and_evaluate(X_train, y_train, X_test, y_test,
                        label: str,
                        scale_pos_weight: float = 1.0) -> dict:
    """Train an XGBoost model and return a metrics dict."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm    = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "strategy":  label,
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, proba), 4),
        # Business cost: FN (missed fraud) costs 10× more than FP (false alarm)
        "business_cost": int(fn * 100 + fp * 10),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "model": model
    }

    print(f"\n{'─'*55}")
    print(f"  Strategy: {label}")
    print(f"{'─'*55}")
    print(classification_report(y_test, preds,
                                 target_names=["Legit", "Fraud"],
                                 zero_division=0))
    print(f"  AUC-ROC       : {results['auc_roc']}")
    print(f"  Confusion matrix:\n    TN={tn}  FP={fp}\n    FN={fn}  TP={tp}")
    print(f"  Estimated business cost (FN×$100, FP×$10): ${results['business_cost']:,}")
    return results


# ── Strategy A: SMOTE oversampling ────────────────────
def strategy_smote(X_train, y_train, X_test, y_test,
                   sample_frac: float = SMOTE_SAMPLE_FRAC) -> dict:
    """
    SMOTE creates synthetic minority-class samples.
    For scalability on 590 K rows, we train on a stratified sample.
    sample_frac controls what fraction of training rows are used.
    """
    if sample_frac < 1.0:
        n = int(len(X_train) * sample_frac)
        X_s = X_train.sample(n=n, random_state=42)
        y_s = y_train.loc[X_s.index]
        print(f"SMOTE: sampled {sample_frac*100:.0f}% of train "
              f"({n:,} rows) to limit RAM usage")
    else:
        X_s, y_s = X_train, y_train

    print(f"  Before SMOTE: {X_s.shape}, Fraud rate: {y_s.mean()*100:.2f}%")
    sm = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_s, y_s)
    print(f"  After  SMOTE: {X_res.shape}, Fraud rate: {y_res.mean()*100:.2f}%")

    return train_and_evaluate(X_res, y_res, X_test, y_test,
                               label="SMOTE Oversampling")


# ── Strategy B: Random undersampling ─────────────────
def strategy_undersample(X_train, y_train, X_test, y_test) -> dict:
    """
    Random undersampling of the majority class to achieve 1:1 balance.
    Much more RAM-efficient than SMOTE.
    """
    fraud_idx  = y_train[y_train == 1].index
    legit_idx  = y_train[y_train == 0].index
    # Downsample majority to match minority count
    legit_down = resample(legit_idx, replace=False,
                          n_samples=len(fraud_idx), random_state=42)
    keep       = fraud_idx.union(legit_down)
    X_res      = X_train.loc[keep]
    y_res      = y_train.loc[keep]

    print(f"  Undersampling: {len(fraud_idx):,} fraud + {len(legit_down):,} legit "
          f"= {len(y_res):,} total | Fraud rate: {y_res.mean()*100:.2f}%")

    return train_and_evaluate(X_res, y_res, X_test, y_test,
                               label="Random Undersampling")


# ── Strategy C: Class weighting (scale_pos_weight) ────
def strategy_class_weight(X_train, y_train, X_test, y_test) -> dict:
    """
    XGBoost scale_pos_weight = #negatives / #positives.
    No resampling — trains on full data. Most efficient strategy.
    """
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Class weighting: scale_pos_weight = {ratio:.2f}")
    return train_and_evaluate(X_train, y_train, X_test, y_test,
                               label="Class Weighting (scale_pos_weight)",
                               scale_pos_weight=ratio)


# ── Comparison & reporting ────────────────────────────
def compare_strategies(results: list[dict]) -> dict:
    print("\n" + "=" * 65)
    print("  IMBALANCE STRATEGY COMPARISON")
    print("=" * 65)
    header = f"{'Strategy':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7} {'Cost':>10}"
    print(header)
    print("-" * 65)
    for r in results:
        print(f"{r['strategy']:<35} {r['precision']:>6} {r['recall']:>6} "
              f"{r['f1']:>6} {r['auc_roc']:>7} ${r['business_cost']:>9,}")
    print("=" * 65)

    # Primary metric: recall (fraud detection priority)
    best_recall = max(results, key=lambda x: x["recall"])
    # Secondary metric: lowest business cost
    best_cost   = min(results, key=lambda x: x["business_cost"])

    print(f"\n  Best recall  : {best_recall['strategy']} "
          f"(Recall={best_recall['recall']})")
    print(f"  Lowest cost  : {best_cost['strategy']} "
          f"(Cost=${best_cost['business_cost']:,})")

    # Save best-recall model
    print("  Saving best-recall model...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_recall["model"],
                f"{ARTIFACT_DIR}/best_model_imbalance.pkl")
    return best_recall


# ── Main ──────────────────────────────────────────────
def run_imbalance_comparison():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.2f}% | "
          f"Test fraud rate: {y_test.mean()*100:.2f}%")

    del X, y
    gc.collect()

    results = []
    results.append(strategy_smote(X_train, y_train, X_test, y_test))
    results.append(strategy_undersample(X_train, y_train, X_test, y_test))
    results.append(strategy_class_weight(X_train, y_train, X_test, y_test))

    best = compare_strategies(results)

    # Save comparison CSV (exclude model objects)
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "model"}
        for r in results
    ])
    out = f"{ARTIFACT_DIR}/imbalance_comparison.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Comparison saved → {out}")
    print(summary.to_string(index=False))
    return best


if __name__ == "__main__":
    run_imbalance_comparison()
