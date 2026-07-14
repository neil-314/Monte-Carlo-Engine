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
    S0 = 100
    K = 105
    T = 1
    r = 0.04
    sigma = 0.20

    call = black_scholes_price(S0, K, T, r, sigma, "call")
    put = black_scholes_price(S0, K, T, r, sigma, "put")
    parity_value = S0 - K * np.exp(-r * T)

    assert abs((call - put) - parity_value) < 1e-10


def test_monte_carlo_option_price_converges_near_black_scholes() -> None:
    S0 = 100
    K = 105
    T = 1
    r = 0.04
    sigma = 0.20

    mc_price = monte_carlo_option_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        simulations=200_000,
        option_type="call",
        seed=2026,
    )
    bs_price = black_scholes_price(S0, K, T, r, sigma, "call")

    assert abs(mc_price - bs_price) < 0.20


def test_var_and_cvar_use_lower_tail() -> None:
    returns = np.array([-0.30, -0.20, -0.10, 0.05, 0.15])

    var = calculate_var(returns, confidence_level=0.80)
    cvar = calculate_cvar(returns, confidence_level=0.80)

    assert np.isclose(var, -0.22)
    assert np.isclose(cvar, -0.30)


def test_portfolio_simulation_outputs_risk_report() -> None:
    portfolio = simulate_correlated_portfolio(
        initial_value=100_000,
        weights=np.array([0.40, 0.30, 0.20, 0.10]),
        expected_returns=np.array([0.08, 0.10, 0.12, 0.06]),
        volatilities=np.array([0.20, 0.25, 0.30, 0.15]),
        correlation_matrix=np.array(
            [
                [1.00, 0.70, 0.55, 0.30],
                [0.70, 1.00, 0.65, 0.25],
                [0.55, 0.65, 1.00, 0.20],
                [0.30, 0.25, 0.20, 1.00],
            ]
        ),
        simulations=5_000,
        seed=99,
        asset_names=["Asset A", "Asset B", "Asset C", "Asset D"],
    )

    report = portfolio_risk_report(
        portfolio["Portfolio Value"].to_numpy(),
        portfolio["Portfolio Return"].to_numpy(),
