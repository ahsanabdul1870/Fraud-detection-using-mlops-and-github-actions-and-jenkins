import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt

ARTIFACT_DIR = os.environ.get(
    "OUTPUT_PATH",
    os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts")
)

def run_explainability():
    print("=== Model Explainability with SHAP ===")
    
    # 1. Load the trained LightGBM model
    model_path = f"{ARTIFACT_DIR}/lgb_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    # 2. Load a sample of the test data to explain
    data_path = f"{ARTIFACT_DIR}/test_features.csv"
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        return
        
    print("Loading a 1000-row sample for SHAP analysis to conserve RAM...")
    X_test = pd.read_csv(data_path, nrows=1000)
    
    if 'isFraud' in X_test.columns:
        y_test = X_test['isFraud']
        X_test = X_test.drop(columns=['isFraud'])
        
    # Ensure columns match model
    expected_features = model.feature_name()
    X_test = X_test[expected_features]
    
    # 3. Calculate SHAP values
    print("Calculating SHAP values...")
    # For LightGBM Booster objects, TreeExplainer is fast
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # LightGBM binary classification usually returns a list of two arrays or one array.
    # We want the SHAP values for the positive class (fraud).
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_fraud = shap_values[1]
    else:
        shap_values_fraud = shap_values

    os.makedirs(f"{ARTIFACT_DIR}/plots", exist_ok=True)

    # 4. Global Explainability: Feature Importance (Summary Plot)
    print("Generating Global Feature Importance Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_fraud, X_test, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/plots/shap_summary_bar.png")
    plt.close()
    
    # 5. Global Explainability: Impact Direction Plot
    print("Generating Global Impact Direction Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_fraud, X_test, show=False)
    plt.title("Feature Impact on Model Output (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/plots/shap_summary_impact.png")
    plt.close()

    # 6. Local Explainability: Explain a single Fraud prediction
    print("Generating Local Explanation for a specific high-risk transaction...")
    # Find a transaction with a high probability of fraud
    probas = model.predict(X_test)
    high_risk_idx = np.argmax(probas)
    
    # Create force plot for this single transaction
    # We use matplotlib backend for saving to image
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values_fraud[high_risk_idx, :], 
        X_test.iloc[high_risk_idx, :], 
        matplotlib=True,
        show=False
    )
    plt.savefig(f"{ARTIFACT_DIR}/plots/shap_force_plot_local.png")
    plt.close()
    
    print(f"\nAll SHAP plots saved successfully to {ARTIFACT_DIR}/plots/")
    print(f"Top high-risk transaction index: {high_risk_idx} with probability: {probas[high_risk_idx]:.4f}")

if __name__ == "__main__":
    run_explainability()
