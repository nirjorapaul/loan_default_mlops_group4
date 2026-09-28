import os
import time
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://172.17.0.1:5000")
mlflow.set_experiment("loan-default-dataops")

DATA_FILE = "loan_data.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Cannot find {DATA_FILE} in this folder")

df = pd.read_csv(DATA_FILE)

# Basic profiling
total_rows = len(df)
total_columns = df.shape[1]
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

print(df.columns.tolist())
print(df.head())

run_name = f"Loan_Ingestion_{time.strftime('%H:%M:%S')}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("data_source", DATA_FILE)
    mlflow.log_metric("total_rows", total_rows)
    mlflow.log_metric("total_columns", total_columns)
    mlflow.log_metric("missing_cells", missing_cells)
    mlflow.log_metric("duplicate_rows", duplicate_rows)
    mlflow.log_artifact(DATA_FILE)

print(f"Rows: {total_rows}, Columns: {total_columns}")
print(f"Missing cells: {missing_cells}, Duplicate rows: {duplicate_rows}")
