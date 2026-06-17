MQFS: Satellite-Based Crop Derivatives Pricing Engine
An institutional-grade infrastructure for pricing agricultural derivatives based on satellite data, built by the Mediterranean Quantitative Finance Society.
This project bridges the gap between agricultural climate risk and quantitative finance by combining high-frequency satellite ingestion, stochastic calculus, and low-latency execution architecture.
 Architecture & Tech Stack
The engine is built with a hybrid approach, separating the data oracle logic from the heavy computational pricing engine to ensure scalability and performance:
Data & Oracle Layer (Python):
src/oracle/: Automated ingestion pipelines extracting indices (NDVI) from satellite data (Sentinel/Google Earth Engine).
src/quant_layer/: Anomaly detection and volatility matrix extraction for feature engineering.
Pricing Engine (C/C++):
c_core/ & cpp/: Low-latency implementation of the Monte Carlo simulations and Ornstein-Uhlenbeck (OU) stochastic processes. Built in C/C++ to bypass Python's GIL and drastically reduce execution time on massive paths.
src/bindings/: Native wrappers bridging the C++ pricer to the Python environment for seamless API interaction.
Calibration & Research (MATLAB):
matlab/: Scripts for rigorous OU model calibration, parameter optimization, and payoff distribution plotting.
 Core Quantitative Models
Ornstein-Uhlenbeck (OU) Process: Utilized to model the mean-reverting behavior of agricultural indices over time.
Monte Carlo Simulations: Deployed via C++ for high-performance path generation to compute fair premiums under complex parametric payoffs.
Advanced Derivatives: Engine designed to handle non-standard, climate-linked parametric payoffs (e.g., burn analysis).
Getting Started
(Insert here a quick command on how to run your pipeline, e.g., python scripts/run_pipeline.py or how to compile the C++ core via the Makefile)
