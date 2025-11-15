# Makefile for Reproducible Data Science Portfolio
.PHONY: install1 install2 project1 project2 all test run run-api run-app train docker


# Project 1: DataProcessor

# Define venv python and pip for Project 1
PYTHON=.venv/bin/python
PIP=.venv/bin/pip
PYTEST=.venv/bin/pytest

# a. Setup environment and install dependencies for Project 1
install1:
	python -m venv .venv
	$(PIP) install -r requirements1.txt
	@echo "Environment setup and dependencies installed for Project 1."

# b. Run data processing script
run:
	$(PYTHON) src/processor_demo.py

# c. Run tests for Project 1
test: 
	$(PYTEST) -v tests/test_processor.py

# d. Clean cache for Project 1
clean_cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cache cleaned."

# e. Run all for Project 1
project1: install1 run test clean_cache
	@echo "Project 1 (DataProcessor) completed."

# Project 2: Web Deployment & Model Serving

# Define venv python and pip for Project 2
PYTHON2=.venv2/bin/python
PIP2=.venv2/bin/pip
PYTEST2=.venv2/bin/pytest

# a. Setup environment and install dependencies for Project 2
install2:
	python -m venv .venv2
	. .venv2/bin/activate && pip install --upgrade pip
	. .venv2/bin/activate && pip install -r requirements2.txt
	@echo "Environment setup and dependencies installed for Project 2."

# b. Fetch initial data
get-data:
	$(PYTHON2) src/fetch_data.py

# c. Scrape data
scrape:
	$(PYTHON2) src/scrape_books.py

# d. Clean data
clean:
	$(PYTHON2) src/clean_books.py

# e. Train model
train:
	$(PYTHON2) src/classify.py

# f. Run FastAPI service
run-api:
	$(PYTHON2) -m uvicorn src.api:app --reload --port 8000

# g. Run Streamlit web app
run-app:
	$(PYTHON2) -m streamlit run src/app.py

# h. Build Docker image
docker:
	docker build -t myapi .
	docker run -p 8000:8000 myapi

# i. Run until step train Project 2
project2: install2 get-data scrape clean train
	@echo "Project 2 (Web Deployment & Model Serving) completed."

# Run all projects in sequence
all: project1 project2
	@echo "All projects completed."
