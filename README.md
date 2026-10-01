
# Loan Default Prediction – End-to-End MLOps Pipeline

An end-to-end MLOps project for predicting loan defaults using machine learning, with automated workflow orchestration, experiment tracking, API deployment, and monitoring.

---

## 1. Project Overview

Loan default prediction is a machine learning problem where the objective is to predict whether a borrower is likely to default on a loan.

This project implements a complete MLOps workflow that takes the system from data ingestion and preprocessing to model training, experiment tracking, deployment, and monitoring.

The project integrates:

- Data Ingestion
- Data Quality and Transformation
- Machine Learning Model Training
- MLflow Experiment Tracking
- Apache Airflow Workflow Automation
- FastAPI Model Deployment
- Prometheus Monitoring
- Grafana Visualization

### Overall Workflow

```
Raw Data
    ↓
Data Ingestion
    ↓
Data Quality & Transformation
    ↓
Model Training
    ↓
MLflow Experiment Tracking
    ↓
FastAPI Deployment
    ↓
Prometheus Monitoring
    ↓
Grafana Dashboard
````

Apache Airflow is used to automate and orchestrate the workflow.

---

## 2. Project Objectives

The main objectives of this project are:

1. Prepare and validate loan-related data.
2. Train a machine learning model to predict loan default.
3. Track machine learning experiments using MLflow.
4. Automate the workflow using Apache Airflow.
5. Deploy the trained model through a FastAPI application.
6. Monitor prediction requests and API behaviour using Prometheus.
7. Visualize monitoring metrics using Grafana.
8. Integrate all components into an end-to-end MLOps pipeline.

---

# 3. Dataset

The project uses loan-related borrower information for predicting loan default.

Important features include:

* Age
* Income
* Loan Amount
* Credit Score
* Months Employed
* Number of Credit Lines
* Interest Rate
* Loan Term
* Debt-to-Income Ratio
* Mortgage Status
* Dependents
* Co-signer
* Education
* Employment Type
* Marital Status
* Loan Purpose

### Target Variable

```
0 → No Default
1 → Default
```

The data is processed and transformed before being passed to the machine learning model.

---

# 4. Project Architecture

```
                       ┌─────────────────┐
                       │     Raw Data    │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │ Data Ingestion  │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │ Data Quality &  │
                       │ Transformation  │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │ Model Training  │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │     MLflow      │
                       │ Experiment      │
                       │ Tracking        │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │    FastAPI      │
                       │   Deployment    │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │   Prometheus    │
                       │    Monitoring   │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │     Grafana     │
                       │    Dashboard    │
                       └─────────────────┘
```

### Workflow Automation

Apache Airflow is used to orchestrate the machine learning workflow.

---

# 5. Project Structure

```
loan-default-mlops/
│
├── apps/
│   └── main.py
│
├── dags/
│   └── ...
│
├── data/
│   └── ...
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── feature_names.json
│
├── src/
│   └── ...
│
├── monitoring/
│   └── prometheus.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 6. Technologies Used

| Technology     | Purpose                            |
| -------------- | ---------------------------------- |
| Python         | Programming language               |
| Pandas         | Data processing                    |
| Scikit-learn   | Machine learning and preprocessing |
| Joblib         | Saving/loading trained components  |
| FastAPI        | Model API and deployment           |
| Uvicorn        | FastAPI server                     |
| MLflow         | Experiment tracking                |
| Apache Airflow | Workflow orchestration             |
| Prometheus     | Metrics collection                 |
| Grafana        | Monitoring visualization           |
| Docker         | Containerization                   |
| Git/GitHub     | Version control                    |

---

# 7. Machine Learning Pipeline

The machine learning pipeline consists of the following stages:

```
Dataset
   ↓
Data Cleaning
   ↓
Feature Transformation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
```

The trained model and preprocessing components are saved for use by the prediction API.

---

# 8. MLflow Experiment Tracking

MLflow is used to track the machine learning training process.

It allows the project to record and view experiment runs and associated training information.

The MLflow interface can be used to inspect the training experiment and its results.

---

# 9. Airflow Workflow Automation

Apache Airflow is used to automate the machine learning workflow.

The project contains the DAG:

```
loan_default_pipeline
```

Airflow manages the execution of the required pipeline tasks.

The workflow can be monitored through the Airflow web interface.

---

## 10. FastAPI Deployment

The trained model is deployed using FastAPI.

The API provides the following endpoints:

### Home

```
GET /
```

Used to check whether the API is running.

### Prediction

```
POST /predict
```

Used to send borrower information and obtain a loan-default prediction.

### Metrics

```
GET /metrics
```

Used to expose monitoring metrics for Prometheus.

---

# 11. Prediction Flow

```
Input Data
    ↓
FastAPI /predict
    ↓
Create DataFrame
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Feature Alignment
    ↓
Scaling / Preprocessing
    ↓
Trained Model
    ↓
Prediction
```

The API returns:

```json
{
    "prediction": 0,
    "result": "No Default"
}
```

or:

```json
{
    "prediction": 1,
    "result": "Default"
}
```

---

# 12. Running the FastAPI Application

Make sure you are in the project root directory.

Run:

```bash
uvicorn apps.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```
http://localhost:8000
```

---

# 13. FastAPI Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

```
http://localhost:8000/docs
```

From Swagger UI, the `/predict` endpoint can be tested by providing the required loan information.

---

# 14. Monitoring with Prometheus

Prometheus collects metrics from the FastAPI application.

The following metrics are implemented:

### Prediction Requests

```text
prediction_requests_total
```

Counts the total number of prediction requests.

### Predicted Defaults

```text
predicted_defaults_total
```

Counts predictions classified as default.

### Prediction Latency

```text
prediction_latency_seconds
```

Measures prediction processing time.

### Prediction Errors

```
prediction_errors_total
```

Counts prediction/API errors.

The metrics are exposed through:

```
http://localhost:8000/metrics
```

---

# 15. Prometheus Configuration

The Prometheus configuration is stored at:

```
monitoring/prometheus.yml
```

The API is configured as a Prometheus target.

Example configuration:

```yaml
scrape_configs:
  - job_name: "loan-api"
    scrape_interval: 5s
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

Prometheus can be started through Docker Compose.

```bash
docker compose up -d prometheus
```

Prometheus UI:

```
http://localhost:9090
```

The target can be checked from:

```text
Status → Targets
```

The `loan-api` target should be available for scraping when the API is running.

---

# 16. Grafana Monitoring

Grafana is used to visualize the metrics collected by Prometheus.

The dashboard contains the following panels:

1. Total Prediction Requests
2. Requests per Second
3. Average Prediction Latency
4. Predicted Defaults
5. API Errors

Grafana:

```
http://localhost:3000
```

Prometheus is configured as the Grafana data source.

---

# 17. Docker

Docker is used to run supporting services such as Prometheus, Grafana and the Airflow environment.

To start the required services:

```bash
docker compose up -d
```

To check running containers:

```bash
docker compose ps
```

To stop the services:

```bash
docker compose down
```

---

# 18. Airflow

The Airflow interface is available at:

```
http://localhost:8080
```

The project DAG is:

```
loan_default_pipeline
```

Airflow is responsible for workflow orchestration and automated execution of the pipeline.

---

# 19. MLflow

MLflow is used for experiment tracking.

The MLflow interface can be started using:

```bash
mlflow ui
```

The interface is generally available at:

```
http://127.0.0.1:5000
```

The training experiment and runs can be viewed from the MLflow interface.

---

# 20. End-to-End Testing

The complete workflow can be checked using the following sequence:

```
1. Start project services
        ↓
2. Run Airflow pipeline
        ↓
3. Verify model training
        ↓
4. Check MLflow experiment
        ↓
5. Start FastAPI
        ↓
6. Test /predict
        ↓
7. Check /metrics
        ↓
8. Verify Prometheus target
        ↓
9. Open Grafana
        ↓
10. Verify monitoring dashboard
```

### Integration Checklist

* [ ] Airflow pipeline runs successfully
* [ ] Model training completes
* [ ] MLflow run is available
* [ ] FastAPI starts successfully
* [ ] `/predict` works
* [ ] `/metrics` works
* [ ] Prometheus connects to the API
* [ ] `loan-api` target is available
* [ ] Grafana connects to Prometheus
* [ ] Dashboard displays monitoring metrics

---

# 21. Monitoring Architecture

The monitoring workflow is:

```
FastAPI
   │
   │ Exposes metrics
   ↓
/metrics
   │
   ↓
Prometheus
   │
   │ Stores and queries metrics
   ↓
Grafana
   │
   ↓
Monitoring Dashboard
```

This allows the deployed prediction API to be observed through request counts, prediction behaviour, latency and errors.

---

# 22. Team Contributions

### P1 – Data Ingestion

Responsible for data collection/loading and preparing the raw data for further processing.

### P2 – Data Quality & Transformation

Responsible for data validation, cleaning and feature transformation.

### P3 – Machine Learning & MLflow

Responsible for model development, training, evaluation and experiment tracking.

### P4 – Airflow & Inference/Deployment

Responsible for workflow automation, inference and model/API deployment.

### P5 – Monitoring, Integration & Documentation

Responsible for:

* Adding monitoring metrics to FastAPI
* Configuring Prometheus
* Creating the Grafana dashboard
* Testing API monitoring
* End-to-end integration
* Project documentation

---

# 23. P5 Monitoring Metrics

The P5 monitoring layer uses the following metrics:

```
prediction_requests_total
        ↓
Total prediction requests

predicted_defaults_total
        ↓
Total predicted defaults

prediction_latency_seconds
        ↓
Prediction processing latency

prediction_errors_total
        ↓
Total prediction errors
```

These metrics are collected by Prometheus and visualized using Grafana.

---

# 24. Expected Outcome

The final system provides an integrated machine learning workflow covering:


Data
 ↓
Data Processing
 ↓
Machine Learning
 ↓
Experiment Tracking
 ↓
Workflow Automation
 ↓
API Deployment
 ↓
Monitoring
 ↓
Visualization
```

The project demonstrates how a machine learning model can be developed, deployed and monitored as an end-to-end MLOps system.

---

#25. Conclusion

This project demonstrates an end-to-end MLOps workflow for loan-default prediction.

It combines data preparation, machine learning, experiment tracking, workflow automation, API deployment and monitoring into an integrated system.

The use of Prometheus and Grafana provides visibility into the deployed prediction service, while Airflow and MLflow support automation and experiment tracking.

