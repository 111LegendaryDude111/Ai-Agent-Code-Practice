PYTHON ?= python3
VENV ?= .venv
UV ?= uv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip --disable-pip-version-check
SRC_PATHS := $(CURDIR)/libs/common/src $(CURDIR)/apps/bot/src $(CURDIR)/services/orchestrator/src
TEST_PYTHONPATH := libs/common/src:apps/bot/src:services/orchestrator/src

.PHONY: help venv install pre-commit-install lint format typecheck test run-bot run-orchestrator ci

help:
	@echo "Targets:"
	@echo "  make venv                - Create local virtual environment"
	@echo "  make install             - Install workspace + dev dependencies"
	@echo "  make pre-commit-install  - Install git hooks"
	@echo "  make lint                - Run ruff checks"
	@echo "  make format              - Run format hooks (pre-commit parity)"
	@echo "  make typecheck           - Run mypy"
	@echo "  make test                - Run unit tests"
	@echo "  make run-bot             - Run bot bootstrap"
	@echo "  make run-orchestrator    - Run orchestrator bootstrap"
	@echo "  make ci                  - Lint + typecheck + tests"

venv:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) venv $(VENV); \
	else \
		$(PYTHON) -m venv $(VENV); \
	fi

install: venv
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) sync --all-packages --group dev; \
	else \
		if $(PIP) install -e libs/common -e apps/bot -e services/orchestrator pre-commit ruff black mypy; then \
			echo "Installed dependencies with pip."; \
		else \
			echo "Online install is unavailable. Creating offline source links for local run."; \
			SITE_PACKAGES=$$($(BIN)/python -c 'import site; print(site.getsitepackages()[0])'); \
			printf "%s\n%s\n%s\n" $(SRC_PATHS) > "$$SITE_PACKAGES/interview_assistant_local.pth"; \
		fi; \
	fi

pre-commit-install: install
	@if [ -x "$(BIN)/pre-commit" ]; then \
		$(BIN)/pre-commit install; \
	else \
		echo "pre-commit is not installed in $(VENV). Run make install with internet access."; \
		exit 1; \
	fi

lint:
	@if [ -x "$(BIN)/ruff" ]; then \
		$(BIN)/ruff check .; \
	else \
		echo "ruff is not installed in $(VENV). Run make install with internet access."; \
		exit 1; \
	fi

format:
	@if [ -x "$(BIN)/pre-commit" ]; then \
		for hook in ruff black; do \
			$(BIN)/pre-commit run $$hook --all-files || true; \
		done; \
		for hook in ruff black; do \
			$(BIN)/pre-commit run $$hook --all-files; \
		done; \
	else \
		if [ ! -x "$(BIN)/ruff" ]; then \
			echo "ruff is not installed in $(VENV). Run make install with internet access."; \
			exit 1; \
		fi; \
		if [ ! -x "$(BIN)/black" ]; then \
			echo "black is not installed in $(VENV). Run make install with internet access."; \
			exit 1; \
		fi; \
		$(BIN)/ruff check --fix .; \
		$(BIN)/black .; \
	fi

typecheck:
	@if [ -x "$(BIN)/mypy" ]; then \
		$(BIN)/mypy .; \
	else \
		echo "mypy is not installed in $(VENV). Run make install with internet access."; \
		exit 1; \
	fi

test:
	@PYTHONPATH=$(TEST_PYTHONPATH) $(BIN)/python -m unittest discover -s tests -v

run-bot:
	@$(BIN)/python -m interview_bot.main

run-orchestrator:
	@$(BIN)/python -m interview_orchestrator.main

ci: lint typecheck test
