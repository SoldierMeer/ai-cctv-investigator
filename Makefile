.PHONY: dev-api dev-web ingest live-cam verify

PYTHON := $(shell if [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

dev-api:
	$(PYTHON) -m uvicorn apps.api-server.app.main:app --port 8000 --reload

dev-web:
	cd apps/web-dashboard && npm run dev

ingest:
	$(PYTHON) apps/vision-pipeline/src/batch_ingest.py

live-cam:
	$(PYTHON) apps/vision-pipeline/src/live_webcam_daemon.py --source 0

verify:
	bash scripts/verify_all.sh
