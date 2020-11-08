VIRTUALENV=venv
PIP=$(VIRTUALENV)/bin/pip
PYTHON=$(VIRTUALENV)/bin/python

devenv: virtualenv install_modules

virtualenv:
	virtualenv -p python3 $(VIRTUALENV) && $(PIP) install --upgrade pip

install_modules:
	$(PIP) install -r requirements.txt

test: unittest

unittest:
	make start_test_deps
	$(PYTHON) -m pytest test
	make stop_test_deps

run:
	$(PYTHON) server.py

run_local:
	$(PYTHON) -m pytest test
	$(PYTHON) server.py

start_test_deps:
	docker-compose -f test/docker-compose.yml up -d

stop_test_deps:
	docker-compose -f test/docker-compose.yml down
