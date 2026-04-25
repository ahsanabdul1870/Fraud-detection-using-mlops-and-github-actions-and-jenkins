import pandas as pd
import numpy as np
import time
import os
import requests

ARTIFACT_DIR = os.environ.get(
    "OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts")
)

def simulate_drift():
    print("=== Simulating Time-Based Data Drift ===")
    
    # 1. Load data
    filepath = f"{ARTIFACT_DIR}/train_features.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    # Read a sample to save RAM
    df = pd.read_csv(filepath, nrows=100000)
    
    # Simulate time-based split: First 50k is 'earlier', last 50k is 'later'
    # We assume data is sorted by time (TransactionDT)
    split_idx = int(len(df) * 0.5)
    df_early = df.iloc[:split_idx].copy()
    df_late = df.iloc[split_idx:].copy()
    
    print(f"Early Data (Train): {df_early.shape}")
    print(f"Late Data (Test): {df_late.shape}")
    
    # 2. Introduce new fraud patterns in 'late' data
    print("Introducing new fraud patterns in 'late' distribution...")
    
    # Pattern A: Sudden spike in large transactions from specific (fake) locations
    # Increase TransactionAmt by 10x for a subset of fraud cases in the late dataset
    fraud_indices = df_late[df_late['isFraud'] == 1].index
    num_to_alter = int(len(fraud_indices) * 0.3) # Alter 30% of frauds
    altered_idx = np.random.choice(fraud_indices, num_to_alter, replace=False)
    
    if 'TransactionAmt_log' in df_late.columns:
        df_late.loc[altered_idx, 'TransactionAmt_log'] += 2.0 # Log scale increase
        
    if 'card1_enc' in df_late.columns:
        # Simulate a specific card type becoming highly fraudulent
        df_late.loc[altered_idx, 'card1_enc'] = 0.99
        
    print(f"Altered {num_to_alter} fraud cases to simulate feature importance shift.")
    
    # Save simulated drift datasets
    df_early.to_csv(f"{ARTIFACT_DIR}/drift_sim_train.csv", index=False)
    df_late.to_csv(f"{ARTIFACT_DIR}/drift_sim_test.csv", index=False)
    
    print(f"Saved drift datasets to {ARTIFACT_DIR}/")
    
    # 3. Send actual traffic to the API to populate Grafana!
    print("Sending live transactions to the API to populate the Grafana dashboard...")
    try:
        # Sample 100 transactions from the early and late sets
        sample_df = pd.concat([df_early.sample(50), df_late.sample(50)])
        
        # We need to send them one by one to simulate real traffic
        for _, row in sample_df.iterrows():
            features_dict = row.to_dict()
            # Send to prediction API
            requests.post("http://localhost:8000/predict", json={"features": features_dict})
            time.sleep(0.05) # Small delay to make the graph look like real traffic
            
        print("Successfully sent 100 transactions to the API.")
        
        # Trigger the Data Drift Alert gauge in Prometheus
        print("Pushing data drift alert metric to Prometheus...")
        requests.get("http://localhost:8000/trigger_drift")
        print("Metric 'data_drift_alert=1' triggered.")
        
        # Triggering GitHub Action repository dispatch (Mocked)
        print("Triggering GitHub Action Retraining via Webhook...")
        # requests.post("https://api.github.com/repos/user/repo/dispatches", json={"event_type": "data_drift_alert"})
        
    except Exception as e:
        print(f"Failed to communicate with API: {e}")

if __name__ == "__main__":
    simulate_drift()
