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
        steps=4,
        simulations=3,
        seed=123,
    )

    expected_path = 100 * np.exp(0.08 * np.linspace(0, 1, 5))
    expected_paths = np.repeat(expected_path[:, np.newaxis], 3, axis=1)

    assert np.allclose(paths, expected_paths)


def test_black_scholes_put_call_parity() -> None:
