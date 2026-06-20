import numpy as np


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate return-based Value at Risk as the lower-tail percentile."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    returns = np.asarray(returns)
    return float(np.percentile(returns, (1 - confidence_level) * 100))


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate return-based CVaR as the average return beyond VaR."""
