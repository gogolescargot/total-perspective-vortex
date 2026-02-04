SHELL := /bin/bash
PY := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: all venv requirements clean fclean re lint

all: lint venv requirements
	@echo 'To activate the virtual environment, run: source $(VENV)/bin/activate'

lint:
	@echo "Running flake8..."
	@$(PY) -m flake8 --exclude=$(VENV); rc=$$?; \
	if [ $$rc -eq 0 ]; then \
		echo "flake8: no issues"; \
	fi; \
	true
	@echo

venv:
	@echo "Checking for virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PY) -m venv $(VENV); \
		echo "Virtual environnement created in $(VENV)"; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@echo

requirements: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo

clean:
	@echo "Cleaning up..."
	-rm -f model.joblib
	-find . -type d -name __pycache__ -exec rm -rf {} +
	@echo

fclean: clean
	@echo "Removing virtual environment..."
	-rm -rf $(VENV)

re: fclean all