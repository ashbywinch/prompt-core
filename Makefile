# Makefile for prompt-core test automation
.PHONY: help setup test test-verbose evals evals-verbose test-unit test-all coverage lint format clean

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
	@echo "  ${GREEN}make test-verbose${NC}  - Run unit tests with verbose output"
	@echo "  ${GREEN}make evals${NC}        - Run evaluation tests with real API (requires API key)"
	@echo "  ${GREEN}make evals-verbose${NC} - Run evaluation tests with verbose output"
	@echo "  ${GREEN}make test-all${NC}      - Run unit tests + evals"
	@echo "  ${GREEN}make coverage${NC}      - Run tests with coverage report"
	@echo "  ${GREEN}make lint${NC}          - Run code linting (black + ruff)"
	@echo "  ${GREEN}make format${NC}        - Auto-fix linting issues"
	@echo "  ${GREEN}make clean${NC}         - Clean up generated files"

# Setup: Install uv if not present, then sync dependencies
setup:
	@uv --version >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv sync --all-extras

run:
	source .venv/bin/activate

test: setup lint test-unit

test-unit: setup
	@${UNITTEST} discover tests/unit/

test-verbose: setup lint
	@${UNITTEST} discover tests/unit/ -v

evals: setup lint
	@${PYTHON} scripts/run_with_timeout.py --timeout 300 -- ${UNITTEST} discover tests/evals/

evals-verbose: setup lint
	@${PYTHON} scripts/run_with_timeout.py --timeout 300 -- ${UNITTEST} discover tests/evals/ -v

test-all: test evals

# Test with coverage (requires coverage package)
coverage: setup
	@${COVERAGE} run -m unittest discover tests/unit/
	@${COVERAGE} report -m
	@${COVERAGE} html
	@echo "${GREEN}Coverage report generated: htmlcov/index.html${NC}"

# Linting with black and ruff
lint: setup
	.venv/bin/black --check --target-version py312 prompt_core/ tests/
	.venv/bin/ruff check prompt_core/ tests/

# Auto-fix linting issues
format: setup
	.venv/bin/black --target-version py312 prompt_core/ tests/
	.venv/bin/ruff check --fix prompt_core/ tests/

# Clean up generated files
clean:
	@rm -rf htmlcov/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f test-results.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
