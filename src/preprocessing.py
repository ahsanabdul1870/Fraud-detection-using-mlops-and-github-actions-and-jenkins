import pandas as pd
import numpy as np
import joblib
import os
import gc

DATA_PATH   = os.environ.get("DATA_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/data"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts"))

# High-cardinality cols: will be TARGET-ENCODED then dropped as strings
HIGH_CARD_COLS = ["card1", "card2", "card3", "card5",
                  "addr1", "addr2", "P_emaildomain", "R_emaildomain"]

DROP_THRESHOLD   = 0.5    # drop column if >50% missing
FLAG_THRESHOLD   = 0.30   # add _was_missing flag if >30% missing


def reduce_mem(df):
    """Downcast numeric dtypes to save RAM."""
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ── 1. Load & Merge ───────────────────────────────────
def load_and_merge(split="train"):
    print(f"Loading {split} data...")
    tx  = reduce_mem(pd.read_csv(f"{DATA_PATH}/{split}_transaction.csv"))
    id_ = reduce_mem(pd.read_csv(f"{DATA_PATH}/{split}_identity.csv"))
    print(f"  tx={tx.shape}, id={id_.shape}")
    df  = tx.merge(id_, on="TransactionID", how="left")
    del tx, id_
    gc.collect()
    df  = reduce_mem(df)
    print(f"  Merged: {df.shape} | RAM: {df.memory_usage().sum()/1e6:.1f} MB")
    return df


# ── 2. Drop high-missing columns ──────────────────────
def drop_high_missing(df, is_train=True):
    """
    BUG FIX: Train and test must drop the SAME columns.
    Train computes which columns to drop and saves the list.
    Test loads that list and applies it — never computes independently.
    """
    if is_train:
        rates     = df.isnull().mean()
        drop_cols = rates[rates > DROP_THRESHOLD].index.tolist()
        # Never drop target or ID
        drop_cols = [c for c in drop_cols
                     if c not in ("isFraud", "TransactionID")]
        joblib.dump(drop_cols, f"{OUTPUT_PATH}/drop_cols.pkl")
        print(f"  Dropped {len(drop_cols)} columns (>{DROP_THRESHOLD*100:.0f}% missing) — saved list")
    else:
        drop_cols = joblib.load(f"{OUTPUT_PATH}/drop_cols.pkl")
        # Only drop cols that actually exist in test (some may already be absent)
        drop_cols = [c for c in drop_cols if c in df.columns]
        print(f"  Dropped {len(drop_cols)} columns (same set as train)")

    df = df.drop(columns=drop_cols)
    return df


# ── 3. Missing value — tiered strategy ────────────────
def handle_missing(df, is_train=True):
    """
    Tiered imputation:
      • >30% missing  → binary _was_missing flag (computed on TRAIN cols, reused on test)
      • All numeric   → median impute (medians from train)
      • Categorical   → fill with UNKNOWN
    """
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ("isFraud", "TransactionID")]
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if is_train:
        rates     = df[num_cols].isnull().mean()
        flag_cols = rates[rates > FLAG_THRESHOLD].index.tolist()
        joblib.dump(flag_cols, f"{OUTPUT_PATH}/flag_cols.pkl")   # ← save for test
    else:
        flag_cols = joblib.load(f"{OUTPUT_PATH}/flag_cols.pkl")
        flag_cols = [c for c in flag_cols if c in df.columns]    # guard

    # Add binary missing-indicator flags
    if flag_cols:
        missing_flags = {
            f"{col}_was_missing": df[col].isnull().astype(np.uint8)
            for col in flag_cols
        }
        df = pd.concat([df, pd.DataFrame(missing_flags, index=df.index)], axis=1)
    print(f"  Added {len(flag_cols)} _was_missing indicator columns")

    # Median imputation — always derived from training data
    if is_train:
        medians = df[num_cols].median()
        joblib.dump(medians, f"{OUTPUT_PATH}/medians.pkl")
    else:
        medians = joblib.load(f"{OUTPUT_PATH}/medians.pkl")
        # Align: only fill cols present in both
        num_cols = [c for c in num_cols if c in medians.index]

    df[num_cols] = df[num_cols].fillna(medians[num_cols])

    # Categorical → UNKNOWN
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    print(f"  Imputed {len(num_cols)} numeric cols | "
          f"Filled {len(cat_cols)} cat cols with UNKNOWN")
    return df


# ── 4. Target encoding for high-cardinality cols ──────
def encode_high_cardinality(df, is_train=True):
    """
    Replace each high-cardinality category with its mean fraud rate.
    Maps computed ONLY on training data — applied to test to prevent leakage.
    Original string columns are dropped after encoding.
    """
    if is_train:
        encoding_maps = {}
        global_mean   = df["isFraud"].mean()
        new_cols = {}
        for col in HIGH_CARD_COLS:
            if col not in df.columns:
                continue
            means              = df.groupby(col)["isFraud"].mean()
            encoding_maps[col] = means.to_dict()
            new_cols[f"{col}_enc"] = (df[col].map(means)
                                      .fillna(global_mean)
                                      .astype(np.float32))
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        joblib.dump(encoding_maps, f"{OUTPUT_PATH}/encoding_maps.pkl")
        joblib.dump(global_mean,   f"{OUTPUT_PATH}/global_mean.pkl")
        print(f"  Target-encoded {len(encoding_maps)} high-card columns")
    else:
        encoding_maps = joblib.load(f"{OUTPUT_PATH}/encoding_maps.pkl")
        global_mean   = joblib.load(f"{OUTPUT_PATH}/global_mean.pkl")
        new_cols = {}
        for col in HIGH_CARD_COLS:
            if col not in df.columns:
                continue
            new_cols[f"{col}_enc"] = (
                df[col].map(encoding_maps.get(col, {}))
                       .fillna(global_mean)
                       .astype(np.float32)
            )
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Drop original high-card string columns (both train and test)
    drop = [c for c in HIGH_CARD_COLS if c in df.columns]
    df   = df.drop(columns=drop)
    return df


# ── 5. Label encode remaining categoricals ────────────
def encode_categoricals(df, is_train=True):
    """
    Label-encode any remaining object columns.
    Train fits & saves encoders; test loads them.
    Unseen categories on test are mapped to -1 (safe sentinel).

    BUG FIX: We save the list of columns encoded in train so that
    test uses exactly those columns — never encodes extra ones.
    """
    if is_train:
        cat_cols = [c for c in df.select_dtypes(include="object").columns
                    if c != "TransactionID"]
        encoders = {}
        for col in cat_cols:
            from sklearn.preprocessing import LabelEncoder
            le           = LabelEncoder()
            df[col]      = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders,  f"{OUTPUT_PATH}/label_encoders.pkl")
        joblib.dump(cat_cols,  f"{OUTPUT_PATH}/label_encoder_cols.pkl")  # ← save col list
        print(f"  Label-encoded {len(cat_cols)} remaining cat cols")
    else:
        from sklearn.preprocessing import LabelEncoder
        encoders = joblib.load(f"{OUTPUT_PATH}/label_encoders.pkl")
        cat_cols = joblib.load(f"{OUTPUT_PATH}/label_encoder_cols.pkl")
        cat_cols = [c for c in cat_cols if c in df.columns]  # guard missing
        for col in cat_cols:
            if col in encoders:
                le      = encoders[col]
                known   = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: int(le.transform([x])[0]) if x in known else -1
                )
        # Drop any remaining object cols NOT in the training cat list
        extra_obj = [c for c in df.select_dtypes(include="object").columns
                     if c != "TransactionID"]
        if extra_obj:
            df = df.drop(columns=extra_obj)
            print(f"  Dropped {len(extra_obj)} extra object cols not seen in train: {extra_obj}")
    return df


# ── 6. Align test columns to match train ──────────────
def align_columns(df, is_train=True):
    """
    Final safety step: ensure test has exactly the same feature columns as
    train (same order). Missing cols are filled with 0; extra cols are dropped.
    This guarantees model input shape is always consistent.
    """
    if is_train:
        cols = [c for c in df.columns if c != "isFraud"]
        joblib.dump(cols, f"{OUTPUT_PATH}/train_feature_cols.pkl")
        print(f"  Saved {len(cols)} train feature column names")
        return df
    else:
        train_cols = joblib.load(f"{OUTPUT_PATH}/train_feature_cols.pkl")
        # Add missing columns as 0
        for c in train_cols:
            if c not in df.columns and c != "isFraud":
                df[c] = 0
        # Keep only train cols (+ isFraud if present)
        keep = [c for c in train_cols if c in df.columns]
        if "isFraud" in df.columns:
            keep = keep  # isFraud not in train_cols (excluded above)
        df = df[keep]
        print(f"  Aligned test to {len(keep)} train columns")
        return df


# ── 7. Full pipeline ──────────────────────────────────
def run_preprocessing(split="train"):
    is_train = (split == "train")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df = load_and_merge(split)
    df = drop_high_missing(df, is_train=is_train)
    df = handle_missing(df, is_train=is_train)
    df = encode_high_cardinality(df, is_train=is_train)
    df = encode_categoricals(df, is_train=is_train)
    df = align_columns(df, is_train=is_train)
    gc.collect()

    out = f"{OUTPUT_PATH}/{split}_preprocessed.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    print(f"  Final shape: {df.shape} | "
          f"RAM: {df.memory_usage().sum()/1e6:.1f} MB")
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print("=== Preprocessing: TRAIN ===")
    train_df = run_preprocessing("train")
    del train_df
    gc.collect()
    print("\n=== Preprocessing: TEST ===")
    test_df = run_preprocessing("test")