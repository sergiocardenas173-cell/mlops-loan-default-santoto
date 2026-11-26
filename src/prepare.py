"""
Etapa 1: Preparacion de datos.
Carga el dataset crudo de Loan Default, selecciona y normaliza las columnas
definidas en el contrato de datos y genera particiones estratificadas 80/20.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "raw", "train.csv")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_OUT_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_OUT_PATH = os.path.join(PROCESSED_DIR, "test.csv")

COLUMN_MAPPING = {
    "ID": "id",
    "Loan Amount": "loan_amount",
    "Funded Amount": "funded_amount",
    "Term": "term",
    "Grade": "grade",
    "Employment Duration": "employment_duration",
    "Verification Status": "verification_status",
    "Delinquency - two years": "delinquency_two_years",
    "Inquires - six months": "inquires_six_months",
    "Open Account": "open_account",
    "Public Record": "public_record",
    "Revolving Balance": "revolving_balance",
    "Total Accounts": "total_accounts",
    "Initial List Status": "initial_list_status",
    "Collection 12 months Medical": "collection_12_months_medical",
    "Application Type": "application_type",
    "Last week Pay": "last_week_pay",
    "Total Collection Amount": "total_collection_amount",
    "Total Current Balance": "total_current_balance",
    "Total Revolving Credit Limit": "total_revolving_credit_limit",
    "Loan Status": "loan_status",
}


def main() -> None:
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"No se encontro el archivo crudo en '{RAW_DATA_PATH}'. "
            "Ubica train.csv dentro de data/raw/ antes de ejecutar el pipeline."
        )

    print(f"[prepare] Leyendo datos crudos desde {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, sep=";", low_memory=False)

    missing_cols = [col for col in COLUMN_MAPPING if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el archivo crudo: {missing_cols}")

    df_clean = df[list(COLUMN_MAPPING.keys())].rename(columns=COLUMN_MAPPING).copy()

    # Eliminar duplicados por ID para garantizar integridad antes de dividir
    df_clean = df_clean.drop_duplicates(subset=["id"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean["loan_status"],
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_OUT_PATH, index=False)
    test_df.to_csv(TEST_OUT_PATH, index=False)

    print(f"[prepare] train.csv generado: {len(train_df):,} filas -> {TRAIN_OUT_PATH}")
    print(f"[prepare] test.csv  generado: {len(test_df):,} filas -> {TEST_OUT_PATH}")


if __name__ == "__main__":
    main()
