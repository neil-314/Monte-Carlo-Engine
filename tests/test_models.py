import numpy as np

from option_pricing import black_scholes_price, monte_carlo_option_price
from portfolio import simulate_correlated_portfolio
from risk_metrics import calculate_cvar, calculate_var, portfolio_risk_report
from simulation import simulate_gbm


def test_gbm_zero_volatility_is_deterministic() -> None:
    paths = simulate_gbm(
        S0=100,
        mu=0.08,
        sigma=0.0,
        T=1,
