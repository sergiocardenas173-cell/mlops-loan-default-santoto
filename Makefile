.PHONY: pipeline prepare validate train test clean

PYTHON ?= python

pipeline:
	$(PYTHON) run_pipeline.py

prepare:
	$(PYTHON) src/prepare.py

validate:
	$(PYTHON) src/validate_data.py

train:
	$(PYTHON) src/train.py

test:
	$(PYTHON) -m pytest -v tests/

clean:
	rm -rf data/processed/*.csv models/*.pkl mlruns/ .pytest_cache/
