# Estandarización de Pipelines MLOps y Control de Calidad de Datos (Loan Default)

**Programa:** Especialización en Ciencia de Datos — Universidad Santo Tomás (USTA)  
**Asignatura:** Seminario de Grado II (Capítulo 2 — Apartado 2.7: *Pipeline único y control de calidad básico*)  
**Autor:** Sergio Daniel Cárdenas Sánchez  

---

## 1. Descripción del proyecto

Este repositorio implementa un **pipeline MLOps estandarizado, reproducible y automatizado** para la clasificación de incumplimiento crediticio (*Loan Default*, 67.463 registros). El flujo permite ejecutar preparación de datos, control de calidad sin librerías externas, entrenamiento con registro de experimentos en MLflow y pruebas automatizadas mediante **un solo comando**.

### Principios aplicados
* **Reproducibilidad:** partición estratificada con semilla fija (`random_state=42`), dependencias declaradas en `requirements.txt` y trazabilidad en MLflow.
* **Control de calidad (*Fail-Fast*):** si los datos incumplen el contrato definido en `data_contract.json` (tipos, nulos, rangos, categorías o fuga de datos), el pipeline se detiene inmediatamente y **no entrena el modelo**.
* **Cero librerías externas en validación:** `src/validate_data.py` está desarrollado 100 % con la biblioteca estándar de Python (`csv`, `json`, `math`, `os`, `sys`).

---

## 2. Estructura del repositorio

```text
mlops-loan-default-santoto/
├── data/
│   ├── raw/
│   │   └── train.csv              # Dataset crudo (delimitado por ';')
│   └── processed/                 # Generado por src/prepare.py (train.csv y test.csv)
├── models/                        # Artefacto entrenado (model.pkl)
├── src/
│   ├── __init__.py
│   ├── prepare.py                 # Etapa 1: limpieza y split 80/20 estratificado
│   ├── validate_data.py           # Etapa 2: validación sin librerías externas + control de leakage
│   └── train.py                   # Etapa 3: RandomForest + tracking en MLflow
├── tests/
│   └── test_model.py              # Etapa 4: pruebas automatizadas con pytest
├── data_contract.json             # Contrato oficial de datos (esquema + ejemplos válido/inválido)
├── run_pipeline.py                # Orquestador principal de un solo comando
├── Makefile                       # Alternativa de automatización (make pipeline)
├── requirements.txt               # Dependencias reproducibles
└── README.md                      # Documentación técnica
```

---

## 3. Requisitos previos

* Python 3.11 o superior (probado en Python 3.13 en Windows).
* Git instalado.
* Archivo `train.csv` ubicado en `data/raw/train.csv`.

---

## 4. Reproducción desde cero (paso a paso)

### 4.1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd mlops-loan-default-santoto
```

### 4.2. Crear y activar el entorno virtual

**En Windows (PowerShell):**
```powershell
py -3.13 -m venv .venv
.venv\Scriptsctivate
```
*(Si PowerShell bloquea la activación, ejecuta primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

**En Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4.4. Ejecutar el pipeline completo con un solo comando
```bash
python run_pipeline.py
```
*(También disponible vía Makefile: `make pipeline`)*

### 4.5. Opciones adicionales de ejecución
* **Probar solo validación, entrenamiento y pruebas sin regenerar particiones:**
  ```bash
  python run_pipeline.py --sin-preparar
  ```
* **Visualizar experimentos en MLflow UI:**
  ```bash
  mlflow ui
  ```
  Abrir en el navegador `http://127.0.0.1:5000`.

---

## 5. Contrato de Datos (`data_contract.json`)

El archivo `data_contract.json` establece el esquema oficial de inferencia y entrenamiento:
* **Tipos de datos:** `integer`, `number` y `string`.
* **Nulabilidad:** `nullable: false` en todas las variables críticas.
* **Rangos lógicos y dominios:** mínimos/máximos financieros y listas cerradas de valores permitidos (`grade`, `term`, `employment_duration`, `verification_status`, `loan_status`).
* **Ejemplos oficiales:** incluye `examples.valid_input` (aprobado por el validador) y `examples.invalid_input` (rechazado con reporte detallado de errores).

---

## 6. Control de fuga de datos (*Data Leakage*)

Además de validar cada fila contra `data_contract.json`, `src/validate_data.py` y `tests/test_model.py` verifican que la intersección de identificadores (`id`) entre `data/processed/train.csv` y `data/processed/test.csv` sea exactamente cero, evitando métricas infladas artificialmente por contaminación de datos.
