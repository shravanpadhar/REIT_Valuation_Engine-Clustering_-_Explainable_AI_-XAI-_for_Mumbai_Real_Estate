.PHONY: install data cluster train pipeline test app clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data_pipeline

cluster:
	$(PYTHON) -m src.clustering

train:
	$(PYTHON) -m src.valuation_model

pipeline: data cluster train

test:
	$(PYTHON) -m pytest tests/ -v

app:
	streamlit run app/app.py

clean:
	rm -rf data/processed/*.parquet models/*.joblib .pytest_cache __pycache__ */__pycache__
