.DEFAULT_GOAL := help

.PHONY: check format help install lint test

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package and development dependencies.
	python -m pip install -e ".[dev]"

format: ## Apply Ruff formatting and safe lint fixes.
	python -m ruff check --fix src tests
	python -m ruff format src tests

lint: ## Check source and test style without modifying files.
	python -m ruff check src tests
	python -m ruff format --check src tests

test: ## Run the CPU-light unit and smoke tests.
	python -m pytest

check: lint test ## Run the same checks used by CI.

