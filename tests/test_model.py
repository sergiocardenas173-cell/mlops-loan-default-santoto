"""
Etapa 4: Pruebas automatizadas con pytest.
Verifica existencia de datos procesados, contrato de datos, artefacto del modelo,
consistencia de predicciones y umbrales minimos de desempeno.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.validate_data import validate_record

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_PATH = os.path.join("models", "model.pkl")
CONTRACT_PATH = "data_contract.json"


def test_processed_datasets_exist_and_not_empty():
    assert os.path.exists(TRAIN_PATH), "No existe data/processed/train.csv"
    assert os.path.exists(TEST_PATH), "No existe data/processed/test.csv"
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    assert not train_df.empty and not test_df.empty
    assert "loan_status" in train_df.columns
    assert "loan_status" in test_df.columns


def test_no_data_leakage_between_train_and_test():
    train_ids = set(pd.read_csv(TRAIN_PATH, usecols=["id"])["id"])
    test_ids = set(pd.read_csv(TEST_PATH, usecols=["id"])["id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Fuga de datos detectada entre train y test"


def test_data_contract_examples():
    assert os.path.exists(CONTRACT_PATH), "No existe data_contract.json"
    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)

    valid_errs = validate_record(
        contract["examples"]["valid_input"],
        contract["features"],
        include_target=False,
    )
    invalid_errs = validate_record(
        contract["examples"]["invalid_input"],
        contract["features"],
        include_target=False,
    )
    assert len(valid_errs) == 0, f"El ejemplo valido genero errores: {valid_errs}"
    assert len(invalid_errs) > 0, "El ejemplo invalido deberia ser rechazado"


def test_model_artifact_exists_and_loads():
    assert os.path.exists(MODEL_PATH), "No existe models/model.pkl"
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_inference_on_valid_contract_example():
    model = joblib.load(MODEL_PATH)
    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)
    sample = dict(contract["examples"]["valid_input"])
    sample.pop("id", None)
    sample_df = pd.DataFrame([sample])
    pred = model.predict(sample_df)
    assert len(pred) == 1
    assert pred[0] in (0, 1)


def test_model_performance_thresholds():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["id", "loan_status"])
    y_test = test_df["loan_status"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    assert acc >= 0.65, f"Accuracy insuficiente: {acc:.4f}"
    assert auc >= 0.45, f"ROC-AUC fuera de rango esperado: {auc:.4f}"
