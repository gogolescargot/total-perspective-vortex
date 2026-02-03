SHELL := /bin/bash
PY := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: all venv clean fclean re

all: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

venv:
	@if [ ! -d "$(VENV)" ]; then \
		$(PY) -m venv $(VENV); \
		echo "Virtual environnement created in $(VENV)"; \
	else \
		echo "$(VENV) already exists"; \
	fi

clean:
	-rm -f data/data_training.csv data/data_validation.csv
	-rm -rf model
	-find . -type d -name __pycache__ -exec rm -rf {} +

fclean: clean
	-rm -rf $(VENV)

re: fclean all