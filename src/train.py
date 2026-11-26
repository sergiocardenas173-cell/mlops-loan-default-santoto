"""
Etapa 3: Entrenamiento del modelo y registro de experimentos con MLflow.
Utiliza un Pipeline de scikit-learn (preprocesamiento + RandomForestClassifier
con balanceo de clases) y registra parametros, metricas y modelo en MLflow.
"""

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

CATEGORICAL_COLS = [
    "grade",
    "employment_duration",
    "verification_status",
    "initial_list_status",
    "application_type",
]

NUMERIC_COLS = [
    "loan_amount",
    "funded_amount",
    "term",
    "delinquency_two_years",
    "inquires_six_months",
    "open_account",
    "public_record",
    "revolving_balance",
    "total_accounts",
    "collection_12_months_medical",
    "last_week_pay",
    "total_collection_amount",
    "total_current_balance",
    "total_revolving_credit_limit",
]


def build_pipeline(n_estimators: int = 100, max_depth: int = 12) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def main() -> None:
    print("[train] Cargando conjuntos de entrenamiento y prueba...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X_train = train_df[feature_cols]
    y_train = train_df["loan_status"]
    X_test = test_df[feature_cols]
    y_test = test_df["loan_status"]

    n_estimators = 100
    max_depth = 12

    pipeline = build_pipeline(n_estimators=n_estimators, max_depth=max_depth)
    print("[train] Entrenando RandomForestClassifier (class_weight='balanced')...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Seminario_Grado_II_Loan_Default")
        with mlflow.start_run(run_name="rf_balanced_pipeline"):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(pipeline, "loan_default_model", serialization_format="cloudpickle")
            print("[train] Experimento registrado correctamente en MLflow.")
    else:
        print("[train] Aviso: mlflow no esta instalado en este entorno; se omitio el tracking.")

    print(f"[train] Modelo guardado en: {MODEL_PATH}")
    print(f"[train] Accuracy: {accuracy:.4f} | F1-macro: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()

