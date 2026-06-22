import numpy as np


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate return-based Value at Risk as the lower-tail percentile."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    returns = np.asarray(returns)
    return float(np.percentile(returns, (1 - confidence_level) * 100))


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate return-based CVaR as the average return beyond VaR."""
    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= var]

    if tail_returns.size == 0:
        return var

    return float(np.mean(tail_returns))


def portfolio_risk_report(
    portfolio_values: np.ndarray,
    portfolio_returns: np.ndarray,
    initial_value: float,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Calculate high-level risk statistics from simulated portfolio outcomes."""
    portfolio_values = np.asarray(portfolio_values)
    portfolio_returns = np.asarray(portfolio_returns)
    var_return = calculate_var(portfolio_returns, confidence_level)
    cvar_return = calculate_cvar(portfolio_returns, confidence_level)

    return {
        "initial_value": float(initial_value),
