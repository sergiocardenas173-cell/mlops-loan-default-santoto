"""
Orquestador principal del pipeline MLOps.
Ejecuta en secuencia estricta:
  1. Preparacion de datos (src/prepare.py)
  2. Validacion y control de calidad (src/validate_data.py)
  3. Entrenamiento y registro con MLflow (src/train.py)
  4. Pruebas automatizadas con pytest (tests/test_model.py)

Aplica el principio "fallar rapido": si cualquier etapa retorna un codigo != 0,
el pipeline se detiene inmediatamente.
"""

import argparse
import subprocess
import sys
import time


def run_stage(stage_name: str, command: list) -> None:
    print("\n" + "=" * 70)
    print(f"INICIANDO {stage_name}")
    print(f"Comando: {' '.join(command)}")
    print("=" * 70)

    start = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Fallo en {stage_name} (codigo de salida: {result.returncode}).")
        print("El pipeline se detiene por principio de fallar rapido.")
        sys.exit(result.returncode)

    print(f"[OK] {stage_name} completada en {elapsed:.2f} s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline MLOps completo de Loan Default con un solo comando."
    )
    parser.add_argument(
        "--sin-preparar",
        action="store_true",
        help="Omite la Etapa 1 (prepare.py) para probar validacion sobre datos ya generados.",
    )
    args = parser.parse_args()

    py = sys.executable
    stages = []

    if not args.sin_preparar:
        stages.append(("ETAPA 1: Preparacion de datos", [py, "src/prepare.py"]))

    stages.extend(
        [
            ("ETAPA 2: Validacion y control de calidad", [py, "src/validate_data.py"]),
            ("ETAPA 3: Entrenamiento y registro con MLflow", [py, "src/train.py"]),
            ("ETAPA 4: Pruebas automatizadas (pytest)", [py, "-m", "pytest", "-v", "tests/"]),
        ]
    )

    total_start = time.time()
    for name, cmd in stages:
        run_stage(name, cmd)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"PIPELINE FINALIZADO CORRECTAMENTE EN {total_elapsed:.2f} s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
