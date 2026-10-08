export PYTHONPATH := src:.

.PHONY: install data notebook train test serve redteam bench eval eval-baseline sweep up down

install:
	pip install -r requirements-dev.txt -r requirements-eval.txt && python -m spacy download en_core_web_lg
data:
	python -c "from common.data import ensure_dataset; print(ensure_dataset())"
notebook:
	python scripts/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/01_causal_and_stats.ipynb
train:
	python -m model_service.train
test:
	pytest -q tests
serve:
	uvicorn model_service.api:app --port 8001 & uvicorn analytics_service.app:app --port 8002 & \
	uvicorn rag_service.app:app --port 8003 & uvicorn mcp_server.server:app --port 8000 --workers 2
redteam:
	python redteam/run_redteam.py
bench:
	python loadtest/bench.py --calls 1000 --concurrency 8 --p95-budget-ms 120
eval:
	python eval/run_eval.py --config configs/rag_tuned.yaml --metric judge --threshold 0.90
eval-baseline:
	python eval/run_eval.py --config configs/rag_baseline.yaml --metric judge --threshold 0.0
sweep:
	python eval/sweep.py --judge-top 3
up:
	docker compose up --build -d
down:
	docker compose down
