"""
Etapa 2: Control de calidad y validacion de datos (100% Python estandar).
NO utiliza librerias externas (sin pandas, numpy, pydantic ni great_expectations).

Verifica contra data_contract.json:
1. Existencia de todas las columnas requeridas.
2. Ausencia de valores nulos o vacios.
3. Tipos de datos correctos (integer, number, string).
4. Rangos logicos (minimum / maximum) y dominios permitidos (allowed_values).
5. Ausencia de fuga de datos (data leakage) por ID entre train.csv y test.csv.
6. Verificacion automatica de los ejemplos valid_input e invalid_input del contrato.
"""

import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

CONTRACT_PATH = "data_contract.json"
TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a"}


def load_contract(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[ERROR] No se encontro el contrato de datos en '{path}'.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def parse_and_validate_value(
    col_name: str, raw_value: Any, rules: Dict[str, Any]
) -> Tuple[Any, List[str]]:
    errors: List[str] = []

    if is_null(raw_value):
        if not rules.get("nullable", False):
            errors.append(f"columna '{col_name}': valor nulo o vacio no permitido.")
        return None, errors

    expected_type = rules["type"]
    parsed_value: Any = raw_value

    if expected_type == "integer":
        if isinstance(raw_value, bool):
            errors.append(f"columna '{col_name}': valor booleano '{raw_value}' no es integer.")
            return None, errors
        try:
            str_val = str(raw_value).strip()
            float_val = float(str_val)
            if not float_val.is_integer():
                errors.append(
                    f"columna '{col_name}': el valor '{raw_value}' no corresponde al tipo integer."
                )
                return None, errors
            parsed_value = int(float_val)
        except ValueError:
            errors.append(
                f"columna '{col_name}': el valor '{raw_value}' no corresponde al tipo integer."
            )
            return None, errors

    elif expected_type == "number":
        if isinstance(raw_value, bool):
            errors.append(f"columna '{col_name}': valor booleano '{raw_value}' no es number.")
            return None, errors
        try:
            parsed_value = float(str(raw_value).strip())
            if math.isnan(parsed_value) or math.isinf(parsed_value):
                errors.append(
                    f"columna '{col_name}': el valor '{raw_value}' no es un numero finito."
                )
                return None, errors
        except ValueError:
            errors.append(
                f"columna '{col_name}': el valor '{raw_value}' no corresponde al tipo float/number."
            )
            return None, errors

    elif expected_type == "string":
        parsed_value = str(raw_value).strip()

    # Validacion de rango numerico
    if expected_type in ("integer", "number"):
        if "minimum" in rules and parsed_value < rules["minimum"]:
            errors.append(
                f"columna '{col_name}': valor {parsed_value} por debajo del minimo ({rules['minimum']})."
            )
        if "maximum" in rules and parsed_value > rules["maximum"]:
            errors.append(
                f"columna '{col_name}': valor {parsed_value} supera el maximo ({rules['maximum']})."
            )

    # Validacion de dominio permitido
    if "allowed_values" in rules and parsed_value not in rules["allowed_values"]:
        errors.append(
            f"columna '{col_name}': valor '{parsed_value}' fuera del dominio permitido {rules['allowed_values']}."
        )

    return parsed_value, errors


def validate_record(
    record: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    include_target: bool = True,
    target_col: str = "loan_status",
) -> List[str]:
    errors: List[str] = []
    for col_name, rules in schema.items():
        if not include_target and col_name == target_col:
            continue
        if rules.get("required", True) and col_name not in record:
            errors.append(f"columna requerida faltante: '{col_name}'.")
            continue
        if col_name in record:
            _, col_errors = parse_and_validate_value(col_name, record[col_name], rules)
            errors.extend(col_errors)
    return errors


def validate_csv(
    csv_path: str, contract: Dict[str, Any], max_errors_to_report: int = 15
) -> Tuple[bool, set]:
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontro el archivo '{csv_path}'.")
        return False, set()

    schema = contract["features"]
    id_col = contract.get("id_column", "id")
    required_cols = [c for c, r in schema.items() if r.get("required", True)]

    errors: List[str] = []
    ids_seen = set()
    row_count = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[ERROR] El archivo '{csv_path}' esta vacio o sin encabezados.")
            return False, set()

        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            print(f"[ERROR] Validacion fallida para {csv_path}: faltan columnas {missing_cols}")
            return False, set()

        for row_idx, row in enumerate(reader, start=2):
            row_count += 1
            if id_col in row and not is_null(row[id_col]):
                ids_seen.add(row[id_col].strip())

            row_errors = validate_record(row, schema, include_target=True)
            for err in row_errors:
                errors.append(f"Fila {row_idx}, {err}")
                if len(errors) >= max_errors_to_report:
                    break
            if len(errors) >= max_errors_to_report:
                break

    if row_count == 0:
        print(f"[ERROR] El archivo '{csv_path}' no contiene registros.")
        return False, ids_seen

    if errors:
        print(f"[ERROR] Validacion fallida para {csv_path}:")
        for err in errors:
            print(f"  - {err}")
        return False, ids_seen

    print(f"[OK] {csv_path} validado correctamente ({row_count:,} filas verificadas).")
    return True, ids_seen


def validate_contract_examples(contract: Dict[str, Any]) -> bool:
    schema = contract["features"]
    target_col = contract.get("target_column", "loan_status")
    examples = contract.get("examples", {})

    valid_input = examples.get("valid_input")
    invalid_input = examples.get("invalid_input")

    if not valid_input or not invalid_input:
        print("[ERROR] El contrato JSON debe incluir 'valid_input' e 'invalid_input'.")
        return False

    valid_errors = validate_record(
        valid_input, schema, include_target=False, target_col=target_col
    )
    if valid_errors:
        print("[ERROR] El ejemplo 'valid_input' del contrato fallo la validacion:")
        for err in valid_errors:
            print(f"  - {err}")
        return False

    invalid_errors = validate_record(
        invalid_input, schema, include_target=False, target_col=target_col
    )
    if not invalid_errors:
        print("[ERROR] El ejemplo 'invalid_input' del contrato fue aceptado inesperadamente.")
        return False

    print(
        f"[OK] Contrato JSON verificado: valid_input aprobado; "
        f"invalid_input rechazado con {len(invalid_errors)} errores detectados."
    )
    return True


def main() -> None:
    print("[validate_data] Iniciando control de calidad (Python estandar, sin librerias externas)...")
    contract = load_contract(CONTRACT_PATH)

    # 1. Validar ejemplos del contrato JSON
    if not validate_contract_examples(contract):
        sys.exit(1)

    # 2. Validar train.csv y test.csv
    train_ok, train_ids = validate_csv(TRAIN_PATH, contract)
    test_ok, test_ids = validate_csv(TEST_PATH, contract)

    if not (train_ok and test_ok):
        sys.exit(1)

    # 3. Control de fuga de datos (data leakage) entre train y test
    overlap = train_ids.intersection(test_ids)
    if overlap:
        print(
            f"[ERROR] Fuga de datos detectada: {len(overlap)} IDs de prueba aparecen en entrenamiento."
        )
        sys.exit(1)

    print("[OK] Control de fuga de datos superado: 0 IDs compartidos entre train y test.")
    print("[validate_data] Todos los controles de calidad fueron aprobados.")


if __name__ == "__main__":
    main()
