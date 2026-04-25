import mlflow
from typing import NamedTuple

BASE_IMAGE = "python:3.10"
SKLEARN    = "scikit-learn==1.4.2"
PANDAS     = "pandas==2.2.2"
NUMPY      = "numpy==1.26.4"
JOBLIB     = "joblib==1.4.2"
LGB        = "lightgbm==4.3.0"
XGB        = "xgboost==2.0.3"
IMBLEARN   = "imbalanced-learn==0.12.3"

# ── Step 1: Data Ingestion ────────────────────────────────
def data_ingestion(data_path: str, output_data: str):
    import pandas as pd
    import numpy as np

    def reduce_mem(df):
        for c in df.select_dtypes(include=np.number).columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype(np.float32)
        return df

    tx_path = f"{data_path}/train_transaction.csv"
    id_path = f"{data_path}/train_identity.csv"

    # Keep identity data in memory once, then stream transactions in chunks
    # to avoid a full-frame merge that can exceed pod memory limits.
    id_df = reduce_mem(pd.read_csv(id_path))
    id_df = id_df.set_index("TransactionID")

    total_rows = 0
    output_cols = None
    first_chunk = True

    for tx_chunk in pd.read_csv(tx_path, chunksize=100_000):
        tx_chunk = reduce_mem(tx_chunk)
        merged = tx_chunk.join(id_df, on="TransactionID", how="left")
        merged = reduce_mem(merged)

        total_rows += len(merged)
        output_cols = merged.shape[1]
        merged.to_csv(output_data, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

    print(f"Ingested {total_rows:,} rows | shape=({total_rows}, {output_cols})")

# ── Step 2: Data Validation ───────────────────────────────
def data_validation(input_data: str) -> bool:
    import pandas as pd
    df = pd.read_csv(input_data, nrows=5000)  # sample for speed

    checks = {
        "has_isFraud":        "isFraud" in df.columns,
        "has_TransactionID":  "TransactionID" in df.columns,
        "has_TransactionAmt": "TransactionAmt" in df.columns,
        "no_negative_amt":    df["TransactionAmt"].min() > 0,
        "fraud_col_binary":   df["isFraud"].isin([0, 1]).all(),
    }
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    assert all(checks.values()), "Validation failed — see above"
    print("All validation checks passed")
    return True

# ── Step 3: Preprocessing ─────────────────────────────────
def data_preprocessing(
    input_data: str,
    output_data: str,
    artifacts_path: str,
    is_train: bool = True,
):
    import pandas as pd
    import numpy as np
    import joblib, os, gc
    from sklearn.preprocessing import LabelEncoder

    os.makedirs(artifacts_path, exist_ok=True)

    HIGH_CARD  = ["card1","card2","card3","card5","addr1","addr2",
                  "P_emaildomain","R_emaildomain"]
    DROP_THR, FLAG_THR = 0.5, 0.3

    # Read in chunks to avoid OOMKill
    sample = pd.read_csv(input_data, nrows=0)
    float64_cols = [c for c in sample.columns]

    chunks = []
    for chunk in pd.read_csv(input_data, chunksize=50_000):
        for c in chunk.select_dtypes("float64").columns:
            chunk[c] = chunk[c].astype(np.float32)
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0, ignore_index=True)
    del chunks; gc.collect()

    if is_train:
        rates = df.isnull().mean()
        drop  = [c for c in rates[rates > DROP_THR].index
                 if c not in ("isFraud", "TransactionID")]
        joblib.dump(drop, f"{artifacts_path}/drop_cols.pkl")
    else:
        drop = [c for c in joblib.load(f"{artifacts_path}/drop_cols.pkl")
                if c in df.columns]
    df.drop(columns=drop, inplace=True)

    num_cols = [c for c in df.select_dtypes(np.number).columns
                if c not in ("isFraud", "TransactionID")]
    cat_cols = df.select_dtypes("object").columns.tolist()

    if is_train:
        flag_cols = df[num_cols].isnull().mean()
        flag_cols = flag_cols[flag_cols > FLAG_THR].index.tolist()
        joblib.dump(flag_cols, f"{artifacts_path}/flag_cols.pkl")
    else:
        flag_cols = [c for c in joblib.load(f"{artifacts_path}/flag_cols.pkl")
                     if c in df.columns]

    new_flag_cols = {}
    if flag_cols:
        for c in flag_cols:
            new_flag_cols[f"{c}_was_missing"] = df[c].isnull().astype(np.uint8)

    if is_train:
        meds = df[num_cols].median()
        joblib.dump(meds, f"{artifacts_path}/medians.pkl")
    else:
        meds     = joblib.load(f"{artifacts_path}/medians.pkl")
        num_cols = [c for c in num_cols if c in meds.index]

    df[num_cols] = df[num_cols].fillna(meds[num_cols])
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")

    new_enc_cols = {}
    if is_train:
        enc_maps, gmean = {}, df["isFraud"].mean()
        for col in HIGH_CARD:
            if col not in df.columns: continue
            m = df.groupby(col)["isFraud"].mean()
            enc_maps[col] = m.to_dict()
            new_enc_cols[f"{col}_enc"] = df[col].map(m).fillna(gmean).astype(np.float32)
        joblib.dump(enc_maps, f"{artifacts_path}/encoding_maps.pkl")
        joblib.dump(gmean,    f"{artifacts_path}/global_mean.pkl")
    else:
        enc_maps = joblib.load(f"{artifacts_path}/encoding_maps.pkl")
        gmean    = joblib.load(f"{artifacts_path}/global_mean.pkl")
        for col in HIGH_CARD:
            if col not in df.columns: continue
            new_enc_cols[f"{col}_enc"] = (df[col].map(enc_maps.get(col, {})))

    all_new = {**new_flag_cols, **new_enc_cols}
    if all_new:
        df = pd.concat([df, pd.DataFrame(all_new, index=df.index)], axis=1)
        del all_new; gc.collect()

    df.drop(columns=[c for c in HIGH_CARD if c in df.columns], inplace=True)

    cat_cols = [c for c in df.select_dtypes("object").columns
                if c != "TransactionID"]
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders,  f"{artifacts_path}/label_encoders.pkl")
        joblib.dump(cat_cols,  f"{artifacts_path}/label_encoder_cols.pkl")
        feat_cols = [c for c in df.columns if c != "isFraud"]
        joblib.dump(feat_cols, f"{artifacts_path}/train_feature_cols.pkl")
    else:
        encoders  = joblib.load(f"{artifacts_path}/label_encoders.pkl")
        saved_cat = joblib.load(f"{artifacts_path}/label_encoder_cols.pkl")
        for col in [c for c in saved_cat if c in df.columns]:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x, le=le: int(le.transform([x])[0]) if x in le.classes_ else -1)
        train_cols = joblib.load(f"{artifacts_path}/train_feature_cols.pkl")
        for c in train_cols:
            if c not in df.columns: df[c] = 0
        df = df[[c for c in train_cols if c in df.columns]]

    gc.collect()
    df.to_csv(output_data, index=False)
    print(f"Preprocessed shape: {df.shape}")

# ── Step 4: Feature Engineering ───────────────────────────
def feature_engineering(
    input_data: str,
    output_data: str,
    artifacts_path: str,
    is_train: bool = True,
):
    import pandas as pd
    import numpy as np
    import joblib, gc, os

    os.makedirs(artifacts_path, exist_ok=True)

    freq_cols = ["card4", "card6"]
    header_cols = pd.read_csv(input_data, nrows=0).columns.tolist()
    present_freq_cols = [c for c in freq_cols if c in header_cols]

    if is_train:
        freq_maps = {}
        for col in present_freq_cols:
            series = pd.read_csv(input_data, usecols=[col])[col]
            freq_maps[col] = series.value_counts(normalize=True).to_dict()
            del series; gc.collect()
        joblib.dump(freq_maps, f"{artifacts_path}/freq_maps.pkl")
    else:
        if os.path.exists(f"{artifacts_path}/freq_maps.pkl"):
            freq_maps = joblib.load(f"{artifacts_path}/freq_maps.pkl")
        else:
            freq_maps = {}

    first_chunk = True
    total_rows  = 0

    for chunk in pd.read_csv(input_data, chunksize=50_000):
        for c in chunk.select_dtypes("float64").columns:
            chunk[c] = chunk[c].astype(np.float32)

        new_cols = {}

        if "TransactionDT" in chunk.columns:
            dt = chunk["TransactionDT"].values.astype(np.float32)
            h  = (dt % 86400  / 3600).astype(np.float32)
            d  = (dt % 604800 / 86400).astype(np.float32)
            new_cols["hour_of_day"] = h
            new_cols["day_of_week"] = d
            new_cols["hour_sin"]    = np.sin(2 * np.pi * h / 24).astype(np.float32)
            new_cols["hour_cos"]    = np.cos(2 * np.pi * h / 24).astype(np.float32)
            new_cols["day_sin"]     = np.sin(2 * np.pi * d / 7).astype(np.float32)
            new_cols["day_cos"]     = np.cos(2 * np.pi * d / 7).astype(np.float32)
            del dt, h, d

        if "TransactionAmt" in chunk.columns:
            amt = chunk["TransactionAmt"].values.astype(np.float32)
            new_cols["TransactionAmt_log"] = np.log1p(amt).astype(np.float32)
            new_cols["TransactionAmt_bucket"] = pd.cut(
                chunk["TransactionAmt"],
                bins=[0, 50, 200, 500, 2000, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(np.float32).values
            del amt

        for col in present_freq_cols:
            if col in freq_maps:
                new_cols[f"{col}_freq"] = (chunk[col]
                                        .map(freq_maps[col])
                                        .fillna(0)
                                        .astype(np.float32))

        if "TransactionAmt_log" in new_cols and "card1_enc" in chunk.columns:
            new_cols["amt_x_card1"] = (new_cols["TransactionAmt_log"] *
                                    chunk["card1_enc"].values).astype(np.float32)

        d_cols = [c for c in chunk.columns if c.startswith("D") and c[1:].isdigit()]
        if len(d_cols) >= 2:
            d_mat = chunk[d_cols].values.astype(np.float32)
            new_cols["D_mean"] = np.nanmean(d_mat, axis=1).astype(np.float32)
            new_cols["D_std"]  = np.nanstd(d_mat,  axis=1).astype(np.float32)
            new_cols["D_max"]  = np.nanmax(d_mat,  axis=1).astype(np.float32)
            del d_mat

        c_cols = [c for c in chunk.columns if c.startswith("C") and c[1:].isdigit()]
        if len(c_cols) >= 2:
            c_mat = chunk[c_cols].values.astype(np.float32)
            new_cols["C_sum"]  = np.nansum(c_mat,  axis=1).astype(np.float32)
            new_cols["C_mean"] = np.nanmean(c_mat, axis=1).astype(np.float32)
            new_cols["C_max"]  = np.nanmax(c_mat,  axis=1).astype(np.float32)
            del c_mat

        v_cols = [c for c in chunk.columns if c.startswith("V") and c[1:].isdigit()]
        if len(v_cols) >= 2:
            v_mat = chunk[v_cols].values.astype(np.float32)
            new_cols["V_mean"] = np.nanmean(v_mat, axis=1).astype(np.float32)
            new_cols["V_std"]  = np.nanstd(v_mat,  axis=1).astype(np.float32)
            del v_mat
            
        if new_cols:
            chunk = pd.concat([chunk, pd.DataFrame(new_cols, index=chunk.index)], axis=1)
            del new_cols

        chunk.drop(
            columns=[c for c in ["TransactionID", "TransactionDT"]
                     if c in chunk.columns],
            inplace=True
        )

        total_rows += len(chunk)
        chunk.to_csv(
            output_data,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False
        del chunk; gc.collect()

    print(f"Feature engineering done: {total_rows:,} rows written")

# ── Step 5: Model Training ──────────────────────────────────
def model_training(
    input_data:      str,
    output_model:    str,
    artifacts_path:  str,
    xgb_sample_frac: float = 0.05,
    max_rows:        int = 120_000,
    neg_to_pos_ratio: int = 4,
):
    import pandas as pd, numpy as np, joblib, gc, os, tempfile, shutil
    from sklearn.model_selection import train_test_split

    os.makedirs(artifacts_path, exist_ok=True)

    print("[1/7] Inferring dtypes...", flush=True)
    sample = pd.read_csv(input_data, nrows=100)
    dtype_dict = {c: np.float32 for c in sample.select_dtypes("float64").columns}
    if "isFraud" in sample.columns:
        dtype_dict["isFraud"] = np.int8
    del sample

    print("[2/7] Counting classes with a lightweight pass...", flush=True)
    pos_total = 0
    neg_total = 0
    for y_chunk in pd.read_csv(input_data, usecols=["isFraud"], chunksize=200_000):
        vals = y_chunk["isFraud"].values
        pos_total += int((vals == 1).sum())
        neg_total += int((vals == 0).sum())

    if pos_total == 0:
        raise ValueError("No positive class rows found in training data.")

    target_total = max(20_000, int(max_rows))
    if pos_total >= target_total:
        target_neg = 0
        pos_keep_prob = min(1.0, float(target_total) / float(max(1, pos_total)))
        neg_keep_prob = 0.0
    else:
        target_neg = min(neg_total, max(1, target_total - pos_total), int(pos_total * max(1, neg_to_pos_ratio)))
        pos_keep_prob = 1.0
        neg_keep_prob = min(1.0, float(target_neg) / float(max(1, neg_total)))

    print("[3/7] Sampling rows in chunks to cap memory...", flush=True)
    sample_dir = tempfile.mkdtemp(prefix="fraud_sample_")
    sampled_csv = f"{sample_dir}/sampled.csv"
    wrote_header = False

    for chunk in pd.read_csv(input_data, dtype=dtype_dict, chunksize=50_000):
        pos = chunk[chunk["isFraud"] == 1]
        neg = chunk[chunk["isFraud"] == 0]

        if pos_keep_prob < 1.0 and len(pos) > 0:
            pos = pos.sample(frac=pos_keep_prob, random_state=42)

        if neg_keep_prob < 1.0 and len(neg) > 0:
            neg = neg.sample(frac=neg_keep_prob, random_state=42)

        sampled_chunk = pd.concat([pos, neg], axis=0, ignore_index=True)
        if len(sampled_chunk) > 0:
            sampled_chunk.to_csv(
                sampled_csv,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
        del pos, neg, sampled_chunk, chunk
        gc.collect()

    if not wrote_header:
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise ValueError("Sampling produced no rows.")

    df = pd.read_csv(sampled_csv, dtype=dtype_dict)
    shutil.rmtree(sample_dir, ignore_errors=True)
    gc.collect()

    if len(df) > target_total:
        df = df.sample(n=target_total, random_state=42, replace=False)

    y            = df["isFraud"].values.astype(np.int8)
    X            = df.drop(columns=["isFraud"]).values.astype(np.float32)
    feature_names = [c for c in df.columns if c != "isFraud"]
    del df; gc.collect()

    print("[4/7] Splitting data...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ratio = float((y_tr == 0).sum()) / float(max(1, (y_tr == 1).sum()))
    del X, y; gc.collect()

    print("[5/7] Spilling splits to disk to free RAM...", flush=True)
    tmp_dir = tempfile.mkdtemp(prefix="fraud_train_")
    np.save(f"{tmp_dir}/X_tr.npy", X_tr)
    np.save(f"{tmp_dir}/X_te.npy", X_te)
    np.save(f"{tmp_dir}/y_tr.npy", y_tr)
    np.save(f"{tmp_dir}/y_te.npy", y_te)
    del X_tr, X_te, y_tr, y_te; gc.collect()

    print("[6/7] Training XGBoost...", flush=True)
    def _train_xgb():
        import xgboost as xgb
        X_tr_mm = np.load(f"{tmp_dir}/X_tr.npy", mmap_mode='r')
        y_tr_mm = np.load(f"{tmp_dir}/y_tr.npy", mmap_mode='r')
        X_te_mm = np.load(f"{tmp_dir}/X_te.npy", mmap_mode='r')
        y_te_mm = np.load(f"{tmp_dir}/y_te.npy", mmap_mode='r')

        rng = np.random.default_rng(42)
        xgb_n = int(len(X_tr_mm) * xgb_sample_frac)
        xgb_n = min(len(X_tr_mm), max(5_000, xgb_n))
        idx = rng.choice(len(X_tr_mm), xgb_n, replace=False)

        X_samp = np.ascontiguousarray(X_tr_mm[idx], dtype=np.float32)
        y_samp = np.ascontiguousarray(y_tr_mm[idx], dtype=np.float32)
        X_te_c = np.ascontiguousarray(X_te_mm,      dtype=np.float32)
        y_te_c = np.ascontiguousarray(y_te_mm,      dtype=np.float32)
        del X_tr_mm, y_tr_mm, X_te_mm, y_te_mm, idx; gc.collect()

        dtrain = xgb.DMatrix(X_samp, label=y_samp, feature_names=feature_names)
        dtest  = xgb.DMatrix(X_te_c, label=y_te_c, feature_names=feature_names)
        del X_samp, y_samp, X_te_c, y_te_c; gc.collect()

        model = xgb.train(
            {
                "objective":        "binary:logistic",
                "eval_metric":      "aucpr",
                "tree_method":      "hist",
                "max_bin":          64,
                "scale_pos_weight": ratio * 2,
                "learning_rate":    0.05,
                "max_depth":        4,
                "nthread":          2,
            },
            dtrain, num_boost_round=100,
            evals=[(dtest, "eval")],
            early_stopping_rounds=15,
            verbose_eval=False,
        )
        joblib.dump(model, f"{artifacts_path}/xgb_model_cost_sens.pkl")
        del dtrain, dtest, model; gc.collect()

    _train_xgb()
    del _train_xgb; gc.collect()

    print("[7/7] Training LightGBM...", flush=True)
    def _train_lgb():
        import lightgbm as lgb
        X_tr_mm = np.load(f"{tmp_dir}/X_tr.npy", mmap_mode='r')
        y_tr_mm = np.load(f"{tmp_dir}/y_tr.npy", mmap_mode='r')
        X_te_mm = np.load(f"{tmp_dir}/X_te.npy", mmap_mode='r')
        y_te_mm = np.load(f"{tmp_dir}/y_te.npy", mmap_mode='r')

        lgb_tr = lgb.Dataset(X_tr_mm, label=y_tr_mm,
                             feature_name=feature_names, free_raw_data=True)
        lgb_te = lgb.Dataset(X_te_mm, label=y_te_mm,
                             reference=lgb_tr, free_raw_data=True)
        del X_tr_mm, y_tr_mm, X_te_mm, y_te_mm; gc.collect()

        model = lgb.train(
            {
                "objective":        "binary",
                "metric":           "auc",
                "scale_pos_weight": ratio * 2,
                "learning_rate":    0.05,
                "num_leaves":       15,
                "max_bin":          63,
                "feature_fraction": 0.8,
                "num_threads":      2,
            },
            lgb_tr, num_boost_round=150,
            valid_sets=[lgb_te],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(-1)],
        )
        joblib.dump(model, f"{artifacts_path}/lgb_model_cost_sens.pkl")
        joblib.dump(model, output_model)
        del lgb_tr, lgb_te, model; gc.collect()

    _train_lgb()
    del _train_lgb; gc.collect()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.", flush=True)

# ── Step 6: Model Evaluation ──────────────────────────────
def model_evaluation(
    input_data:  str,
    input_model: str,
) -> tuple:
    import pandas as pd, numpy as np, joblib
    from sklearn.metrics import (roc_auc_score, recall_score,
                                 precision_score, f1_score, confusion_matrix)

    model = joblib.load(input_model)

    y_parts = []
    proba_parts = []
    preds_parts = []
    for chunk in pd.read_csv(input_data, chunksize=50_000):
        y_chunk = chunk["isFraud"].to_numpy(dtype=np.int8)
        X_chunk = chunk.drop(columns=["isFraud"]).select_dtypes(include="number")
        proba_chunk = model.predict(X_chunk)
        preds_chunk = (proba_chunk > 0.5).astype(np.int8)

        y_parts.append(y_chunk)
        proba_parts.append(proba_chunk)
        preds_parts.append(preds_chunk)

    y = np.concatenate(y_parts)
    proba = np.concatenate(proba_parts)
    preds = np.concatenate(preds_parts)

    auc       = roc_auc_score(y, proba)
    recall    = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    f1        = f1_score(y, preds, zero_division=0)
    cm        = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    cost = int(fn * 100 + fp * 10)

    import mlflow
    mlflow.log_metric("auc_roc",       float(auc))
    mlflow.log_metric("recall",        float(recall))
    mlflow.log_metric("precision",     float(precision))
    mlflow.log_metric("f1",            float(f1))
    mlflow.log_metric("business_cost", float(cost))
    mlflow.log_metric("TP", int(tp)); mlflow.log_metric("FP", int(fp))
    mlflow.log_metric("FN", int(fn)); mlflow.log_metric("TN", int(tn))

    print(f"AUC={auc:.4f} | Recall={recall:.4f} | F1={f1:.4f} | Cost=${cost:,}")
    return float(auc), float(recall)

# ── Step 7: Conditional Deployment ───────────────────────
def conditional_deploy(
    auc_roc:        float,
    recall:         float,
    auc_threshold:  float = 0.90,
    recall_threshold: float = 0.80,
):
    auc_ok    = auc_roc >= auc_threshold
    recall_ok = recall  >= recall_threshold
    if auc_ok and recall_ok:
        print("Both thresholds passed — DEPLOYING model")
    else:
        print("Threshold(s) not met — skipping deployment")

# ── MLflow Run ───────────────────────────────────────────
def run_pipeline(data_path="data", artifacts_path="fraud-artifacts"):
    import os
    os.makedirs(artifacts_path, exist_ok=True)
    
    ingest_out = f"{artifacts_path}/ingested_data.csv"
    prep_out = f"{artifacts_path}/preprocessed_data.csv"
    feat_out = f"{artifacts_path}/engineered_data.csv"
    model_out = f"{artifacts_path}/final_model.pkl"

    import mlflow
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run(run_name="fraud-detection-mlflow"):
        print("Starting Data Ingestion...")
        data_ingestion(data_path, ingest_out)
        
        print("Starting Data Validation...")
        data_validation(ingest_out)
        
        print("Starting Data Preprocessing...")
        data_preprocessing(ingest_out, prep_out, artifacts_path, True)
        
        print("Starting Feature Engineering...")
        feature_engineering(prep_out, feat_out, artifacts_path, True)
        
        print("Starting Model Training...")
        model_training(feat_out, model_out, artifacts_path, xgb_sample_frac=0.05)
        
        print("Starting Model Evaluation...")
        auc, recall = model_evaluation(feat_out, model_out)
        
        print("Conditional Deployment Check...")
        conditional_deploy(auc, recall)
        
        mlflow.log_artifact(model_out, "models")
        print("Pipeline Complete. Run tracked in MLflow.")

if __name__ == "__main__":
    run_pipeline("/home/ahsan/Videos/mlops/assignment3/data", "/home/ahsan/Videos/mlops/assignment3/fraud-artifacts")
