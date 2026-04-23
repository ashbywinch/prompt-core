# Makefile for prompt-core test automation
.PHONY: help test test-unit test-basic test-integration test-all coverage lint clean

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

all: 
	uv sync
help:
	@echo "Available commands:"
	@echo "  ${GREEN}make${NC}          	 - Get project ready to run"
	@echo "  ${GREEN}make help${NC}          - Show this help message"
	@echo "  ${GREEN}make test${NC}          - Run all unit tests (basic + conversation)"
	@echo "  ${GREEN}make test-unit${NC}     - Run all unit tests under tests/unit/"
	@echo "  ${GREEN}make test-basic${NC}    - Run basic model tests only (no mocking)"
	@echo "  ${GREEN}make test-conversation${NC} - Run conversation orchestrator tests"
	@echo "  ${GREEN}make test-integration${NC} - Run integration tests with real API (requires OPENAI_API_KEY)"
	@echo "  ${GREEN}make test-all${NC}      - Run ALL tests (unit + integration)"
	@echo "  ${GREEN}make coverage${NC}      - Run tests with coverage report"
	@echo "  ${GREEN}make lint${NC}          - Run code linting (if available)"
	@echo "  ${GREEN}make clean${NC}         - Clean up generated files"

test: test-basic test-conversation
	@echo "${GREEN}✓ All unit tests passed${NC}"

test-unit:
	@echo "${YELLOW}Running all unit tests...${NC}"
	@${PYTHON} test_conversation.py

test-basic:
	@echo "${YELLOW}Running basic model tests...${NC}"
	@${PYTHON} test_basic.py

test-conversation: test-basic
	@echo "${YELLOW}Running orchestrator and LLM interaction tests...${NC}"
	@${UNITTEST} tests/unit/test_orchestrator_logic.py -v
	@${UNITTEST} tests/unit/test_llm_interaction.py -v

test-integration:
	@echo "${YELLOW}Running integration tests (requires OPENAI_API_KEY)...${NC}"
	@echo "${YELLOW}Note: These tests will ${RED}FAIL${YELLOW} if API key is missing${NC}"
	@${UNITTEST} discover tests/integration/ -v

test-all: test test-integration
	@echo "${GREEN}✓ All tests (unit + integration) completed${NC}"
	@echo "${YELLOW}Note: Integration tests may fail if OPENAI_API_KEY is not set${NC}"

# Test with coverage (requires coverage package)
coverage:
	@if [ ! -f .venv/bin/coverage ]; then \
		echo "${YELLOW}Installing coverage package...${NC}"; \
		.venv/bin/pip install coverage; \
	fi
	@echo "${YELLOW}Running tests with coverage...${NC}"
	@${COVERAGE} run -m unittest discover tests/unit/
	@${COVERAGE} report -m
	@${COVERAGE} html
	@echo "${GREEN}Coverage report generated: htmlcov/index.html${NC}"

# Linting (if you have flake8, black, or other linters)
lint:
	@if [ -f .venv/bin/flake8 ]; then \
		echo "${YELLOW}Running flake8...${NC}"; \
		.venv/bin/flake8 prompt_core/ tests/; \
	else \
		echo "${YELLOW}flake8 not installed. Install with: pip install flake8${NC}"; \
	fi
	@if [ -f .venv/bin/black ]; then \
		echo "${YELLOW}Running black check...${NC}"; \
		.venv/bin/black --check prompt_core/ tests/; \
	else \
		echo "${YELLOW}black not installed. Install with: pip install black${NC}"; \
	fi

# Clean up generated files
clean:
	@echo "${YELLOW}Cleaning up...${NC}"
	@rm -rf htmlcov/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f test-results.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "${GREEN}✓ Cleanup complete${NC}"

# Run all tests in CI environment (fails if integration tests can't run)
ci-test: test
	@echo "${GREEN}✓ CI tests passed (unit tests only)${NC}"
	@echo "${YELLOW}Note: Integration tests not run in CI by default${NC}"
	@echo "${YELLOW}To run integration tests: make test-integration (requires OPENAI_API_KEY)${NC}"
