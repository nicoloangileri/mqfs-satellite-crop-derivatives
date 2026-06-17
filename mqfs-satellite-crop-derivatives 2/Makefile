# =====================================================================
# MQFS Satellite Crop Derivatives — top-level Makefile
# =====================================================================
# Common entry points:
#   make native     build the C and C++ shared libraries
#   make test       run the pytest suite (28 tests)
#   make figures    run the pipeline -> figures/ + paper/generated/
#   make paper      compile the working paper to PDF (needs pdflatex)
#   make all        native + test + figures + paper
#   make clean      remove build artefacts
# =====================================================================

PYTHON ?= python3
CC     ?= gcc
CXX    ?= g++
CFLAGS   = -O3 -std=c99 -Wall -fPIC
CXXFLAGS = -O3 -std=c++17 -Wall -fPIC -fopenmp

ROOT := $(shell pwd)

.PHONY: all native c cpp test figures paper clean

all: native test figures paper

# --- native engines --------------------------------------------------
native: c cpp

c:
	$(CC) $(CFLAGS) -shared -o c_core/libmqfs_ou_mc.so c_core/ou_mc.c -lm
	@echo "built c_core/libmqfs_ou_mc.so"

cpp:
	$(CXX) $(CXXFLAGS) -shared -o cpp/libmqfs_cpp.so cpp/capi.cpp
	@echo "built cpp/libmqfs_cpp.so"

# Optional CLIs (handy for a quick sanity check)
cli: c cpp
	$(CC)  $(CFLAGS)   -o c_core/cli_demo c_core/cli_demo.c c_core/ou_mc.c -lm
	$(CXX) $(CXXFLAGS) -o cpp/cli cpp/cli.cpp

# --- tests -----------------------------------------------------------
test: native
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

# --- pipeline: figures + generated LaTeX tables ----------------------
figures: native
	$(PYTHON) scripts/run_pipeline.py

# --- paper -----------------------------------------------------------
paper: figures
	cd paper && pdflatex -interaction=nonstopmode mqfs_wp_satellite_crop.tex >/dev/null \
	         && pdflatex -interaction=nonstopmode mqfs_wp_satellite_crop.tex >/dev/null
	@echo "built paper/mqfs_wp_satellite_crop.pdf"

# --- cleanup ---------------------------------------------------------
clean:
	rm -f c_core/*.so cpp/*.so c_core/cli_demo cpp/cli
	rm -f paper/*.aux paper/*.log paper/*.out
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "cleaned"
