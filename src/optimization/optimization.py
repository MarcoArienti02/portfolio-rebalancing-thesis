import numpy as np
import pandas as pd

def covariance_matrix(returns: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    data = returns.drop(columns=[date_col], errors="ignore").dropna()
    return data.cov()


def _as_cov_array(cov: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    names = list(cov.columns) if isinstance(cov, pd.DataFrame) else None
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("The covariance matrix must be square.")
    return (arr + arr.T) / 2, names

# Confirm that a fully invested portfolio is feasible under the bounds.
def _check_bounds(n: int, min_weight: float, max_weight: float) -> None:
    if min_weight * n > 1 or max_weight * n < 1:
        raise ValueError("Weight bounds do not permit a fully invested portfolio.")


def _scipy_minimize(*args, **kwargs):
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SciPy is required for portfolio optimization.") from exc
    return minimize(*args, **kwargs)

# Format optimized weights with asset names when available.
def _format_weights(weights: np.ndarray, names: list[str] | None) -> pd.Series:
    return pd.Series(weights, index=names) if names is not None else pd.Series(weights)

# Remove numerical noise and renormalize weights.
def _clean_solution(weights: np.ndarray) -> np.ndarray:
    weights = np.where(np.abs(weights) < 1e-12, 0, weights)
    return weights / weights.sum()

# Global Minimum Variance weights minimize w'Σw.
def gmv_weights(
    cov: pd.DataFrame | np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    cov_arr, names = _as_cov_array(cov)
    n = cov_arr.shape[0]
    _check_bounds(n, min_weight, max_weight)

    # Long-only GMV with weights summing to one.
    res = _scipy_minimize(
        lambda w: w @ cov_arr @ w,
        np.repeat(1 / n, n),
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"GMV optimization failed: {res.message}")
    return _format_weights(_clean_solution(res.x), names)

# Normalized risk contribution of each asset.
def risk_contributions(weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray) -> pd.Series:
    cov_arr, names = _as_cov_array(cov)
    w = np.asarray(weights, dtype=float)
    port_var = w @ cov_arr @ w
    rc = w * (cov_arr @ w) / port_var
    return _format_weights(rc, names)

# Risk Parity portfolio weights.
def risk_parity_weights(
    cov: pd.DataFrame | np.ndarray,
    risk_budget: np.ndarray | pd.Series | None = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    cov_arr, names = _as_cov_array(cov)
    n = cov_arr.shape[0]
    _check_bounds(n, min_weight, max_weight)
    # Build and normalize the risk budget.
    budget = np.repeat(1 / n, n) if risk_budget is None else np.asarray(risk_budget, dtype=float)
    if budget.shape != (n,) or not np.isfinite(budget).all() or (budget <= 0).any():
        raise ValueError("The risk budget must contain one positive finite value per asset.")
    budget = budget / budget.sum()

    # Convex risk-budgeting formulation, scaled for monthly covariance matrices.
    initial_weights = np.repeat(1 / n, n)
    initial_variance = initial_weights @ cov_arr @ initial_weights
    if not np.isfinite(initial_variance) or initial_variance <= 0:
        raise ValueError("The covariance matrix does not imply positive portfolio variance.")
    initial_x = initial_weights / np.sqrt(initial_variance)

    def objective(x: np.ndarray) -> float:
        return 0.5 * (x @ cov_arr @ x) - budget @ np.log(x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return cov_arr @ x - budget / x

    res = _scipy_minimize(
        objective,
        initial_x,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 3000, "maxls": 100},
    )
    if not res.success:
        raise RuntimeError(f"Risk Parity optimization failed: {res.message}")

    weights = _clean_solution(res.x)
    bound_tolerance = 1e-10
    if (weights < min_weight - bound_tolerance).any() or (weights > max_weight + bound_tolerance).any():
        raise RuntimeError("The Risk Parity solution violates the requested weight bounds.")

    # Validate normalized contributions directly rather than trusting only the solver flag.
    port_var = weights @ cov_arr @ weights
    contributions = weights * (cov_arr @ weights) / port_var
    max_contribution_error = np.max(np.abs(contributions - budget))
    contribution_tolerance = 1e-6
    if not np.isfinite(contributions).all() or max_contribution_error > contribution_tolerance:
        raise RuntimeError(
            "Numerically invalid Risk Parity solution: "
            f"maximum contribution error = {max_contribution_error:.3e}, "
            f"tolerance = {contribution_tolerance:.1e}."
        )

    return _format_weights(weights, names)

# Calculate rolling optimized weights month by month.
def rolling_optimized_weights(
    returns: pd.DataFrame,
    method: str,
    window: int = 60,
    date_col: str = "Date",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    data = returns.sort_values(date_col).dropna().reset_index(drop=True)
    asset_cols = data.columns.drop(date_col)
    optimizer = {"gmv": gmv_weights, "risk_parity": risk_parity_weights}[method]
    rows = []

    # Use only prior returns at each date to avoid look-ahead bias.
    for i in range(window, len(data)):
        cov = data.loc[i - window : i - 1, asset_cols].cov()
        weights = optimizer(cov, min_weight=min_weight, max_weight=max_weight)
        rows.append({"Date": data.loc[i, date_col], **weights.to_dict()})

    return pd.DataFrame(rows)
