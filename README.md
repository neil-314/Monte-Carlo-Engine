# Monte Carlo Quantitative Risk & Derivatives Engine

[Live simulator](https://neil-314.github.io/Monte-Carlo-Engine/) | [Source code](https://github.com/neil-314/Monte-Carlo-Engine)

A Python-based quantitative finance project for simulating equity prices, pricing European options, and analyzing portfolio risk using Monte Carlo methods.

This project is intentionally built without live market data. The inputs are hard-coded so the model assumptions, formulas, and outputs are easy to inspect and explain.

## Overview

The engine uses Monte Carlo simulation to model uncertain future stock prices, estimate European option values, validate those estimates against the Black-Scholes analytical model, and simulate a correlated multi-asset portfolio. It includes both a Python analysis engine and a static browser simulator served through GitHub Pages.

## Features

- Geometric Brownian Motion stock-price simulation
- 10,000+ single-stock price paths
- Terminal price statistics and distribution analysis
- European call and put option pricing
- Black-Scholes analytical validation
- Monte Carlo convergence analysis
- Multi-asset portfolio simulation
- Correlated asset returns using Cholesky decomposition
- Value at Risk and CVaR / Expected Shortfall
- Portfolio loss probability and return distribution
- Dark-mode browser simulator in `docs/index.html`
- Basic automated tests
- Jupyter notebook for interactive analysis

## Project Structure

```text
monte-carlo-engine/
|-- README.md
|-- requirements.txt
|-- main.py
|-- docs/
|   |-- index.html
|   `-- .nojekyll
|-- src/
|   |-- __init__.py
|   |-- simulation.py
|   |-- option_pricing.py
|   |-- portfolio.py
|   |-- risk_metrics.py
|   `-- run_analysis.py
|-- notebooks/
|   `-- analysis.ipynb
|-- tests/
|   |-- conftest.py
|   `-- test_models.py
|-- results/
|   |-- stock_paths.png
|   |-- terminal_distribution.png
|   |-- option_convergence.png
|   `-- portfolio_distribution.png
```

## Installation

On Windows, if `python` opens the Microsoft Store, use the Python launcher:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate all reports and charts:

```powershell
.\.venv\Scripts\python.exe main.py
```

Run individual modules:

```powershell
