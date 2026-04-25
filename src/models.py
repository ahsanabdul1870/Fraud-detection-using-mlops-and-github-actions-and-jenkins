"""
models.py — Task 3: Model Complexity + Task 4: Cost-Sensitive Learning

Models trained:
  1. XGBoost  (Standard)       ← Task 3 requirement
  2. XGBoost  (Cost-Sensitive) ← Task 4 requirement
  3. LightGBM (Standard)       ← Task 3 requirement
  4. LightGBM (Cost-Sensitive) ← Task 4 requirement
  5. Hybrid: RF feature selection → LightGBM ← Task 3 hybrid requirement

RAM strategy (dataset: 590K rows × 250 cols ≈ 647 MB as float32):
  • All data kept as float32 (halves RAM vs float64)
  • XGBoost uses tree_method='hist' — histogram-based, same low-RAM
    algorithm as LightGBM; avoids the O(n²) exact-greedy method
  • XGBoost trained on a 50% stratified sample by default
    (configurable via XGB_SAMPLE_FRAC env var) — full data would
    push DMatrix to ~1.3 GB on top of the feature DataFrame
  • Each model's DMatrix / Dataset is deleted + gc.collect() called
    immediately after training before the next model starts
  • RF in hybrid uses only 20% sample + max_depth=5 to cap RAM
  • LightGBM uses lgb.Dataset which is bin-compressed internally
"""

import pandas as pd
import numpy as np
import gc
import os
import joblib
import time

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ARTIFACT_DIR = os.environ.get(
    "OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts")
)

# XGBoost sample fraction — set to 1.0 to train on full data (needs ~2 GB RAM)
XGB_SAMPLE_FRAC = float(os.environ.get("XGB_SAMPLE_FRAC", "0.5"))


# ── RAM Utilities ─────────────────────────────────────────
def reduce_mem(df):
    """Downcast numeric dtypes to save RAM."""
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ── Load Data ─────────────────────────────────────────────
def load_features(split="train", sample_frac=1.0):
    """
    Load feature-engineered CSV efficiently.
    sample_frac < 1.0 draws a stratified sample to reduce RAM.
    """
    print(f"Loading {split} features...")
    filepath = f"{ARTIFACT_DIR}/{split}_features.csv"
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found. "
              "Run preprocessing.py then features.py first.")
        return None, None

    df = reduce_mem(pd.read_csv(filepath))

    if sample_frac < 1.0:
        # Stratified sample to preserve fraud rate
        fraud    = df[df["isFraud"] == 1].sample(frac=sample_frac, random_state=42)
        legit    = df[df["isFraud"] == 0].sample(frac=sample_frac, random_state=42)
        df       = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
        del fraud, legit
        gc.collect()
        print(f"  Stratified sample: {sample_frac*100:.0f}% → {len(df):,} rows")

    y = df["isFraud"].astype(np.int8)
    X = df.drop(columns=["isFraud"])
    del df
    gc.collect()

    print(f"  X={X.shape} | RAM: {X.memory_usage().sum()/1e6:.1f} MB | "
          f"Fraud rate: {y.mean()*100:.2f}%")
    return X, y


# ── Shared Evaluation ─────────────────────────────────────
def evaluate_model(y_true, preds, proba, model_name="Model",
                   cost_fn=100, cost_fp=10):
    """
    Full evaluation: Precision / Recall / F1 / AUC-ROC /
    Confusion matrix / Business cost estimate.
    """
    precision = precision_score(y_true, preds, zero_division=0)
    recall    = recall_score(y_true, preds, zero_division=0)
    f1        = f1_score(y_true, preds, zero_division=0)
    auc       = roc_auc_score(y_true, proba)
    cm        = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * cost_fn + fp * cost_fp

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, preds,
                                 target_names=["Legit", "Fraud"],
                                 zero_division=0))
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={tn:>6}  FP={fp:>6}")
    print(f"    FN={fn:>6}  TP={tp:>6}")
    print(f"  Business cost (FN×${cost_fn}, FP×${cost_fp}): ${total_cost:,}")
    print(f"{'='*50}\n")

    return {
        "model":     model_name,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc_roc":   round(auc, 4),
        "cost":      int(total_cost),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "cm":        cm,
    }


# ── XGBoost ───────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test,
                  cost_sensitive=False,
                  sample_frac=XGB_SAMPLE_FRAC):
    """
    RAM-efficient XGBoost training.

    Key settings:
      tree_method='hist'  — histogram-based splits (same algorithm as LightGBM).
                            Much less RAM than default 'exact' method.
      max_bin=256         — number of histogram bins; lower = less RAM.
      subsample=0.8       — row subsampling per tree (built-in, no extra copy).
      colsample_bytree=0.8 — feature subsampling per tree.

    sample_frac: fraction of TRAINING rows used to build the DMatrix.
      Default 0.5 — trains on ~236K rows instead of 472K, cutting DMatrix
      RAM by half (~300 MB instead of ~600 MB) while retaining accuracy.
      Set XGB_SAMPLE_FRAC=1.0 env var to use full data if RAM allows.
    """
    label = "XGBoost" + (" (Cost-Sensitive)" if cost_sensitive else " (Standard)")
    print(f"\n--- Training {label} ---")

    # Stratified subsample of training data
    if sample_frac < 1.0:
        fraud_idx = y_train[y_train == 1].index
        legit_idx = y_train[y_train == 0].index
        fraud_s   = fraud_idx[np.random.default_rng(42).choice(
                        len(fraud_idx), int(len(fraud_idx) * sample_frac), replace=False)]
        legit_s   = legit_idx[np.random.default_rng(42).choice(
                        len(legit_idx), int(len(legit_idx) * sample_frac), replace=False)]
        idx       = fraud_s.union(legit_s)
        Xs, ys    = X_train.loc[idx], y_train.loc[idx]
        print(f"  XGB sample: {sample_frac*100:.0f}% → {len(ys):,} rows "
              f"(fraud rate: {ys.mean()*100:.2f}%)")
    else:
        Xs, ys = X_train, y_train

    ratio = (ys == 0).sum() / (ys == 1).sum()
    spw   = ratio * 2 if cost_sensitive else ratio  # 2× penalty for cost-sensitive

    # Build DMatrix (XGBoost's optimised internal format — float32 saves RAM)
    dtrain = xgb.DMatrix(Xs.astype(np.float32), label=ys)
    dtest  = xgb.DMatrix(X_test.astype(np.float32), label=y_test)

    # Free source arrays immediately — DMatrix holds its own copy
    if sample_frac < 1.0:
        del Xs, ys
    gc.collect()

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",          # AUCPR better than AUC for imbalance
        "tree_method":      "hist",           # ← RAM-efficient histogram method
        "max_bin":          256,              # histogram bins (lower = less RAM)
        "scale_pos_weight": spw,
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,             # row subsampling (no extra copy)
        "colsample_bytree": 0.8,             # feature subsampling
        "seed":             42,
        "nthread":          -1,
    }

    start = time.time()
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False,
    )
    print(f"  Training took {time.time()-start:.1f}s | "
          f"Best round: {model.best_iteration}")

    proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
    preds = (proba > 0.5).astype(int)

    # Free DMatrix objects immediately
    del dtrain, dtest
    gc.collect()

    metrics = evaluate_model(y_test, preds, proba, model_name=label)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    fname = f"xgb_model{'_cost_sens' if cost_sensitive else ''}.pkl"
    joblib.dump(model, f"{ARTIFACT_DIR}/{fname}")
    print(f"  Saved → {ARTIFACT_DIR}/{fname}")

    return metrics, model


# ── LightGBM ──────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test, cost_sensitive=False):
    """
    LightGBM is already histogram-based and bin-compresses data internally
    via lgb.Dataset — the most RAM-efficient option for this dataset.
    Trains on full data (no subsampling needed).
    """
    label = "LightGBM" + (" (Cost-Sensitive)" if cost_sensitive else " (Standard)")
    print(f"\n--- Training {label} ---")

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    spw   = ratio * 2 if cost_sensitive else ratio

    # lgb.Dataset compresses data into histograms — much smaller than raw arrays
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_eval  = lgb.Dataset(X_test,  y_test,  reference=lgb_train, free_raw_data=True)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "scale_pos_weight": spw,
        "learning_rate":    0.05,
        "num_leaves":       63,       # more leaves = richer model, manageable RAM
        "max_depth":        6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    start = time.time()
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    print(f"  Training took {time.time()-start:.1f}s | "
          f"Best round: {model.best_iteration}")

    # Free Dataset objects
    del lgb_train, lgb_eval
    gc.collect()

    proba = model.predict(X_test, num_iteration=model.best_iteration)
    preds = (proba > 0.5).astype(int)

    metrics = evaluate_model(y_test, preds, proba, model_name=label)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    fname = f"lgb_model{'_cost_sens' if cost_sensitive else ''}.pkl"
    joblib.dump(model, f"{ARTIFACT_DIR}/{fname}")
    print(f"  Saved → {ARTIFACT_DIR}/{fname}")

    return metrics, model


# ── Hybrid: RF Feature Selection → LightGBM ───────────────
def train_hybrid(X_train, y_train, X_test, y_test, n_top_features=30):
    """
    Two-stage hybrid model (Task 3 requirement):
      Stage 1 — RandomForest on a 20% stratified sample → top-N feature importance
      Stage 2 — LightGBM trained only on the top-N selected features

    RAM savings:
      • RF uses only 20% sample + shallow trees (max_depth=5, n_estimators=50)
      • RF is deleted immediately after feature selection
      • LightGBM then trains on a narrow feature matrix (N cols instead of 250)
    """
    print(f"\n--- Training Hybrid Model (RF feature selection → LightGBM) ---")

    # ── Stage 1: RF feature selection on 20% stratified sample ──
    print(f"  Stage 1: RF feature selection on 20% sample...")
    fraud_idx = y_train[y_train == 1].index
    legit_idx = y_train[y_train == 0].index
    rng       = np.random.default_rng(42)
    f_s = fraud_idx[rng.choice(len(fraud_idx), int(len(fraud_idx)*0.2), replace=False)]
    l_s = legit_idx[rng.choice(len(legit_idx), int(len(legit_idx)*0.2), replace=False)]
    idx = f_s.union(l_s)
    X_s, y_s = X_train.loc[idx], y_train.loc[idx]
    print(f"    RF sample: {len(y_s):,} rows | fraud rate: {y_s.mean()*100:.2f}%")

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,        # shallow = less RAM
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_s, y_s)

    top_features = (pd.Series(rf.feature_importances_, index=X_train.columns)
                    .nlargest(n_top_features).index.tolist())
    print(f"    Top {n_top_features} features selected: {top_features[:5]} ...")

    del rf, X_s, y_s
    gc.collect()

    # ── Stage 2: LightGBM on selected features only ─────────
    print(f"  Stage 2: LightGBM on {n_top_features} selected features...")
    ratio     = (y_train == 0).sum() / (y_train == 1).sum()
    lgb_train = lgb.Dataset(X_train[top_features], y_train, free_raw_data=True)
    lgb_eval  = lgb.Dataset(X_test[top_features],  y_test,
                             reference=lgb_train, free_raw_data=True)

    params = {
        "objective":   "binary",
        "metric":      "auc",
        "scale_pos_weight": ratio,
        "learning_rate": 0.05,
        "num_leaves":  31,
        "n_jobs":      -1,
        "verbose":     -1,
    }

    model = lgb.train(
        params, lgb_train, num_boost_round=300,
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    del lgb_train, lgb_eval
    gc.collect()

    proba = model.predict(X_test[top_features],
                          num_iteration=model.best_iteration)
    preds = (proba > 0.5).astype(int)

    metrics = evaluate_model(y_test, preds, proba,
                              model_name="Hybrid (RF→LightGBM)")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": top_features},
                f"{ARTIFACT_DIR}/hybrid_model.pkl")
    print(f"  Saved → {ARTIFACT_DIR}/hybrid_model.pkl")

    return metrics, model


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_features("train", sample_frac=1.0)
    if X is None:
        raise SystemExit("No data found. Run preprocessing.py → features.py first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape} | "
          f"Train fraud rate: {y_train.mean()*100:.2f}%")

    del X, y
    gc.collect()

    results = []

    # ── 1. XGBoost Standard (Task 3) ──────────────────────
    m, _ = train_xgboost(X_train, y_train, X_test, y_test, cost_sensitive=False)
    results.append(m)
    gc.collect()

    # ── 2. XGBoost Cost-Sensitive (Task 4) ────────────────
    m, _ = train_xgboost(X_train, y_train, X_test, y_test, cost_sensitive=True)
    results.append(m)
    gc.collect()

    # ── 3. LightGBM Standard (Task 3) ─────────────────────
    m, _ = train_lightgbm(X_train, y_train, X_test, y_test, cost_sensitive=False)
    results.append(m)
    gc.collect()

    # ── 4. LightGBM Cost-Sensitive (Task 4) ───────────────
    m, _ = train_lightgbm(X_train, y_train, X_test, y_test, cost_sensitive=True)
    results.append(m)
    gc.collect()

    # ── 5. Hybrid: RF → LightGBM (Task 3) ─────────────────
    m, _ = train_hybrid(X_train, y_train, X_test, y_test)
    results.append(m)
    gc.collect()

    # ── Summary ───────────────────────────────────────────
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "cm"}
        for r in results
    ])
    out = f"{ARTIFACT_DIR}/models_comparison.csv"
    summary.to_csv(out, index=False)

    print("\n" + "="*70)
    print("  MODELS COMPARISON SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70)
    print(f"\nSaved → {out}")
