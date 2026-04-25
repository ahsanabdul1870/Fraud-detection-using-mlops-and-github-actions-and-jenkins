import pandas as pd
import numpy as np
import joblib
import os
import gc

OUTPUT_PATH = os.environ.get("OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts"))


# ── 1. Time-based features ────────────────────────────
def add_time_features(df):
    """
    TransactionDT is seconds from a reference point.
    Extract hour-of-day and day-of-week as cyclic (sin/cos) features
    so that time wraps correctly (e.g. 23:00 ≈ 00:00).

    All new columns are batched into a single pd.concat to avoid
    the PerformanceWarning caused by repeated single-column insertions
    on a highly fragmented DataFrame.
    """
    if "TransactionDT" not in df.columns:
        return df

    seconds_in_day  = 86_400
    seconds_in_week = 604_800

    hour = (df["TransactionDT"] % seconds_in_day / 3_600).astype(np.float32)
    day  = (df["TransactionDT"] % seconds_in_week / seconds_in_day).astype(np.float32)

    new_cols = pd.DataFrame({
        "hour_of_day": hour,
        "day_of_week": day,
        "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
        "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
        "day_sin":  np.sin(2 * np.pi * day  / 7).astype(np.float32),
        "day_cos":  np.cos(2 * np.pi * day  / 7).astype(np.float32),
    }, index=df.index)

    df = pd.concat([df, new_cols], axis=1)
    print("  Added time-based features (hour_of_day, day_of_week, cyclic sin/cos)")
    return df


# ── 2. Transaction amount features ────────────────────
def add_amount_features(df):
    """
    Log-transform skewed TransactionAmt to reduce outlier impact.
    Also bucket amount into ordinal bands for tree models.
    New columns batched via pd.concat to prevent fragmentation warnings.
    """
    if "TransactionAmt" not in df.columns:
        return df

    new_cols = pd.DataFrame({
        "TransactionAmt_log": np.log1p(df["TransactionAmt"]).astype(np.float32),
        "TransactionAmt_bucket": pd.cut(
            df["TransactionAmt"],
            bins=[0, 50, 200, 500, 2_000, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(np.float32),
    }, index=df.index)

    df = pd.concat([df, new_cols], axis=1)
    print("  Added amount features (log-transform, 5-band bucket)")
    return df


# ── 3. Frequency encoding ─────────────────────────────
def add_frequency_features(df, is_train=True):
    """
    Frequency encoding: replace category value with how often it appears
    (normalised count). Useful to signal rare vs common cards.

    Note: card1 & card2 are in HIGH_CARD_COLS and are DROPPED as strings
    during preprocessing (only their _enc target-encoded versions survive).
    We therefore only frequency-encode card4 & card6, which are low-cardinality
    and retained as label-encoded integers after preprocessing.
    New columns batched via pd.concat to prevent fragmentation warnings.
    """
    freq_cols = [c for c in ["card4", "card6"] if c in df.columns]

    if not freq_cols:
        print("  No frequency-encoding columns found (card4/card6 missing)")
        return df

    new_cols = {}
    if is_train:
        freq_maps = {}
        for col in freq_cols:
            freq           = df[col].value_counts(normalize=True)
            freq_maps[col] = freq.to_dict()
            new_cols[f"{col}_freq"] = df[col].map(freq).astype(np.float32)
        joblib.dump(freq_maps, f"{OUTPUT_PATH}/freq_maps.pkl")
        print(f"  Frequency-encoded {len(freq_cols)} cols: {freq_cols}")
    else:
        if os.path.exists(f"{OUTPUT_PATH}/freq_maps.pkl"):
            freq_maps = joblib.load(f"{OUTPUT_PATH}/freq_maps.pkl")
            for col in freq_cols:
                if col in freq_maps:
                    new_cols[f"{col}_freq"] = (df[col].map(freq_maps[col])
                                               .fillna(0).astype(np.float32))
            print(f"  Applied frequency encoding for {len(freq_cols)} cols (from train maps)")
        else:
            print("  freq_maps.pkl not found — skipping frequency encoding on test")

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ── 4. Interaction features ───────────────────────────
def add_interaction_features(df):
    """
    Domain-specific interaction features for fraud detection.
      • amt × card1_enc  : unusual amounts on specific cards
      • D-column stats   : time deltas — irregular timing signals fraud
      • C-column stats   : count columns — high sums may signal card abuse
      • V-column stats   : Vesta-engineered features — moment aggregates

    All new columns batched into a single pd.concat call.
    """
    new_cols = {}

    # Amount × target-encoded card1 (card1_enc produced by preprocessing)
    if "TransactionAmt_log" in df.columns and "card1_enc" in df.columns:
        new_cols["amt_x_card1"] = (df["TransactionAmt_log"] *
                                    df["card1_enc"]).astype(np.float32)
        print("  Added amt_x_card1 interaction feature")

    # D-column aggregates (time-delta features)
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    if len(d_cols) >= 2:
        new_cols["D_mean"] = df[d_cols].mean(axis=1).astype(np.float32)
        new_cols["D_std"]  = df[d_cols].std(axis=1).astype(np.float32)
        new_cols["D_max"]  = df[d_cols].max(axis=1).astype(np.float32)
        print(f"  Added D-column aggregates (mean, std, max) from {len(d_cols)} D cols")

    # C-column aggregates (count features)
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    if len(c_cols) >= 2:
        new_cols["C_sum"]  = df[c_cols].sum(axis=1).astype(np.float32)
        new_cols["C_mean"] = df[c_cols].mean(axis=1).astype(np.float32)
        new_cols["C_max"]  = df[c_cols].max(axis=1).astype(np.float32)
        print(f"  Added C-column aggregates (sum, mean, max) from {len(c_cols)} C cols")

    # V-column aggregates (Vesta-engineered features)
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if len(v_cols) >= 2:
        new_cols["V_mean"] = df[v_cols].mean(axis=1).astype(np.float32)
        new_cols["V_std"]  = df[v_cols].std(axis=1).astype(np.float32)
        print(f"  Added V-column aggregates (mean, std) from {len(v_cols)} V cols")

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ── 5. Drop non-feature columns ───────────────────────
def drop_non_features(df):
    """Drop ID and raw datetime columns — not predictive features."""
    drop = [c for c in ["TransactionID", "TransactionDT"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        print(f"  Dropped non-feature cols: {drop}")
    return df


# ── 6. Full feature engineering pipeline ──────────────
def run_feature_engineering(split="train"):
    is_train = (split == "train")
    in_path  = f"{OUTPUT_PATH}/{split}_preprocessed.csv"

    print(f"Loading preprocessed {split} data...")
    df = pd.read_csv(in_path)

    # Downcast floats: CSV load loses float32 precision info
    float64_cols = df.select_dtypes(include="float64").columns
    df[float64_cols] = df[float64_cols].astype(np.float32)

    # Defragment the DataFrame before adding any new columns.
    # Preprocessing did many pd.concat operations which leave the internal
    # block structure fragmented — .copy() consolidates it cleanly.
    df = df.copy()
    print(f"  Shape: {df.shape} | RAM: {df.memory_usage().sum()/1e6:.1f} MB")

    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_frequency_features(df, is_train=is_train)
    df = add_interaction_features(df)
    df = drop_non_features(df)

    # Final defragmentation pass after all concat operations
    df = df.copy()
    gc.collect()

    out = f"{OUTPUT_PATH}/{split}_features.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    print(f"  Final shape: {df.shape} | RAM: {df.memory_usage().sum()/1e6:.1f} MB")
    return df


if __name__ == "__main__":
    print("=== Feature Engineering: TRAIN ===")
    train_df = run_feature_engineering("train")
    del train_df
    gc.collect()
    print("\n=== Feature Engineering: TEST ===")
    test_df = run_feature_engineering("test")
