# Makefile for prompt-core test automation
.PHONY: help setup test evals test-unit test-all coverage lint clean

# Variables
PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
UNITTEST := .venv/bin/python -m unittest
COVERAGE := .venv/bin/coverage

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "Available commands:"
	@echo "  ${GREEN}make${NC}          	 - Get project ready to run (setup + sync)"
	@echo "  ${GREEN}make setup${NC}      - Install uv if not present"
	@echo "  ${GREEN}make help${NC}          - Show this help message"
	@echo "  ${GREEN}make test${NC}          - Run unit tests (no API key required)"
	@echo "  ${GREEN}make evals${NC}        - Run evaluation tests with real API (requires API key)"
	@echo "  ${GREEN}make test-all${NC}      - Run ALL tests (unit + evals)"
	@echo "  ${GREEN}make coverage${NC}      - Run tests with coverage report"
	@echo "  ${GREEN}make lint${NC}          - Run code linting"
	@echo "  ${GREEN}make clean${NC}         - Clean up generated files"

# Setup: Install uv if not present, then sync dependencies
setup:
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv sync --all-extras

test: test-unit

test-unit:
	@${UNITTEST} discover tests/unit/ -v

evals:
	@${UNITTEST} discover tests/evals/ -v

test-all: test-unit evals

# Test with coverage (requires coverage package)
coverage:
	@${COVERAGE} run -m unittest discover tests/unit/
	@${COVERAGE} report -m
	@${COVERAGE} html
	@echo "${GREEN}Coverage report generated: htmlcov/index.html${NC}"

# Linting with black
lint:
	.venv/bin/black --check --target-version py312 prompt_core/ tests/

# Clean up generated files
clean:
	@rm -rf htmlcov/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f test-results.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

