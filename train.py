from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT FOLDERS
# ============================================================

# This file must be saved inside:
# LoanDefaultProject/src/train.py

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42
ROC_AUC_THRESHOLD = 0.75

# LEAVE THIS EMPTY FOR YOUR FIRST LOCAL TEST.
# Later, replace "" with the MLflow URL given by your teacher/team.
MLFLOW_TRACKING_URI = ""


# ============================================================
# 3. MLflow SETUP
# ============================================================

if MLFLOW_TRACKING_URI.strip():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    # Use SQLite database for local MLflow tracking
    mlflow.set_tracking_uri(
        f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    )

mlflow.set_experiment("loan-default-group4")



# ============================================================
# 4. LOAD PROCESSED DATA FROM PERSON 2
# ============================================================

print("\nLoading processed data...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Could not find train.csv here:\n{TRAIN_PATH}\n"
        "Put train.csv inside data/processed/"
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Could not find test.csv here:\n{TEST_PATH}\n"
        "Put test.csv inside data/processed/"
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "Default"

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in train.csv/test.csv."
    )

print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")
print(f"Default distribution in train:\n{train_df[TARGET].value_counts()}\n")


# ============================================================
# 5. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Person 2 already encoded/scaled the data.
# Convert Boolean columns to numeric so every model receives
# a clean numeric matrix.
X_train = X_train.astype(float)
X_test = X_test.astype(float)

print(f"Number of features: {X_train.shape[1]}")
print(f"Training rows    : {X_train.shape[0]}")
print(f"Testing rows     : {X_test.shape[0]}")


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 7. HELPER FUNCTION FOR METRICS + PLOTS
# ============================================================

def evaluate_model(model, model_name):
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # --------------------------------------------------------
    # Confusion matrix image
    # --------------------------------------------------------

    safe_name = model_name.lower().replace(" ", "_")

    cm_path = REPORTS_DIR / f"{safe_name}_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # Feature importance / coefficient plot
    # --------------------------------------------------------

    importance_path = REPORTS_DIR / f"{safe_name}_feature_importance.png"

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        )
    elif hasattr(model, "coef_"):
        importance = pd.Series(
            abs(model.coef_[0]),
            index=X_train.columns,
        )
    else:
        importance = pd.Series(dtype=float)

    if not importance.empty:
        importance = importance.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        importance.sort_values().plot(kind="barh", ax=ax)

        ax.set_title(f"{model_name} - Top 15 Feature Importance")
        ax.set_xlabel("Importance")

        plt.tight_layout()
        plt.savefig(importance_path, dpi=150)
        plt.close()

    return {
        "model": model,
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "confusion_matrix_path": cm_path,
        "importance_path": importance_path,
        "predictions": predictions,
        "probabilities": probabilities,
    }


# ============================================================
# 8. TRAIN ALL MODELS + LOG EACH RUN TO MLflow
# ============================================================

results = []

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name.replace(" ", "_")):

        result = evaluate_model(model, model_name)

        # Log parameters
        model_params = {
            key: value
            for key, value in model.get_params().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }

        mlflow.log_params(model_params)

        # Log metrics
        mlflow.log_metrics({
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "roc_auc": result["roc_auc"],
        })

        # Log plots
        mlflow.log_artifact(str(result["confusion_matrix_path"]))

        if result["importance_path"].exists():
            mlflow.log_artifact(str(result["importance_path"]))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_test.head(3),
            serialization_format="cloudpickle",
        )

        results.append(result)


# ============================================================
# 9. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame([
    {
        "model": r["model_name"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "roc_auc": r["roc_auc"],
    }
    for r in results
])

results_df = results_df.sort_values(
    by=["recall", "roc_auc"],
    ascending=False,
)

results_path = REPORTS_DIR / "model_results.csv"
results_df.to_csv(results_path, index=False)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]

best_result = next(
    r for r in results
    if r["model_name"] == best_model_name
)

print(f"\nSelected model: {best_model_name}")
print(f"Selected model recall : {best_result['recall']:.4f}")
print(f"Selected model ROC-AUC: {best_result['roc_auc']:.4f}")


# ============================================================
# 10. MODEL QUALITY CHECK FOR AIRFLOW
# ============================================================

if best_result["roc_auc"] < ROC_AUC_THRESHOLD:
    raise ValueError(
        f"MODEL QUALITY CHECK FAILED: ROC-AUC = "
        f"{best_result['roc_auc']:.4f}, required >= {ROC_AUC_THRESHOLD}"
    )

print(
    f"\nMODEL QUALITY CHECK PASSED: "
    f"ROC-AUC {best_result['roc_auc']:.4f} >= {ROC_AUC_THRESHOLD}"
)


# ============================================================
# 11. SAVE THE SELECTED MODEL
# ============================================================

best_model = best_result["model"]

model_path = MODELS_DIR / "model.pkl"
joblib.dump(best_model, model_path)

features_path = MODELS_DIR / "feature_names.json"

with open(features_path, "w", encoding="utf-8") as f:
    json.dump(list(X_train.columns), f, indent=2)

selection_info = {
    "selected_model": best_model_name,
    "precision": float(best_result["precision"]),
    "recall": float(best_result["recall"]),
    "f1": float(best_result["f1"]),
    "roc_auc": float(best_result["roc_auc"]),
    "roc_auc_threshold": ROC_AUC_THRESHOLD,
}

selection_path = REPORTS_DIR / "model_selection.json"

with open(selection_path, "w", encoding="utf-8") as f:
    json.dump(selection_info, f, indent=2)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Model saved to        : {model_path}")
print(f"Feature names saved to: {features_path}")
print(f"Results saved to      : {results_path}")
print(f"Selection saved to    : {selection_path}")
print(f"MLflow tracking URI   : {mlflow.get_tracking_uri()}")
print("\nNext step:")
print("Open MLflow UI with:")
print("    mlflow ui")
print("Then open http://127.0.0.1:5000 in your browser.")
