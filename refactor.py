import re

with open("pipeline/fraud_pipeline.py", "r") as f:
    content = f.read()

# Remove kfp imports
content = re.sub(r"import kfp\n", "", content)
content = re.sub(r"from kfp import dsl\n", "import mlflow\n", content)
content = re.sub(r"from kfp\.dsl import.*?\n", "", content)
content = re.sub(r"try:\n    from kfp import kubernetes as kfp_kubernetes\n    _HAS_KFP_K8S = True\nexcept ImportError:\n    _HAS_KFP_K8S = False\n", "", content)

# Remove @component decorators
content = re.sub(r"@component\(.*?\)\n", "", content, flags=re.DOTALL)

# Replace Input[Dataset], Output[Dataset] etc with str
content = re.sub(r"Input\[Dataset\]", "str", content)
content = re.sub(r"Output\[Dataset\]", "str", content)
content = re.sub(r"Input\[Model\]", "str", content)
content = re.sub(r"Output\[Model\]", "str", content)
content = re.sub(r"Output\[Metrics\]", "dict", content)

# Remove NamedTuple
content = re.sub(r"-> NamedTuple.*?:\s*# type: ignore\[valid-type\]", ":", content)

# Replace .path with nothing (since the arg is now just a string path)
content = re.sub(r"output_data\.path", "output_data", content)
content = re.sub(r"input_data\.path", "input_data", content)
content = re.sub(r"output_model\.path", "output_model", content)
content = re.sub(r"input_model\.path", "input_model", content)

# Fix metrics
content = re.sub(r"metrics:\s*dict", "", content) # Remove metrics from args if it exists
content = re.sub(r"metrics\.log_metric\(", "mlflow.log_metric(", content)

# Fix EvalOutputs return in model_evaluation
content = re.sub(r"EvalOutputs = namedtuple.*?\]\)", "", content)
content = re.sub(r"return EvalOutputs\(auc_roc=float\(auc\), recall=float\(recall\)\)", "return float(auc), float(recall)", content)

# Remove @dsl.pipeline block entirely and the pipeline definition
pipeline_def_start = content.find("@dsl.pipeline(")
if pipeline_def_start != -1:
    content = content[:pipeline_def_start]

# Add a new main block for MLflow
main_block = """
def run_pipeline(data_path="/mnt/data", artifacts_path="/mnt/fraud-artifacts"):
    import os
    os.makedirs(artifacts_path, exist_ok=True)
    
    ingest_out = f"{artifacts_path}/ingested_data.csv"
    prep_out = f"{artifacts_path}/preprocessed_data.csv"
    feat_out = f"{artifacts_path}/engineered_data.csv"
    model_out = f"{artifacts_path}/final_model.pkl"

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
        # Since we removed metrics argument, we need to ensure the call matches
        auc, recall = model_evaluation(feat_out, model_out)
        
        print("Conditional Deployment Check...")
        conditional_deploy(auc, recall)
        
        mlflow.log_artifact(model_out, "models")
        print("Pipeline Complete. Run tracked in MLflow.")

if __name__ == "__main__":
    run_pipeline("/home/ahsan/Videos/mlops/assignment3/data", "/home/ahsan/Videos/mlops/assignment3/fraud-artifacts")
"""

content += main_block

# Fix signature of model_evaluation by removing empty comma if present
content = re.sub(r",\s*\)", ")", content)

with open("pipeline/fraud_pipeline.py", "w") as f:
    f.write(content)
print("Done modifying fraud_pipeline.py")
