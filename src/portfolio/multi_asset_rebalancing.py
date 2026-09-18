import numpy as np
import pandas as pd

FREQ_MONTHS = {"M": 1, "Q": 3, "Y": 12, "monthly": 1, "quarterly": 3, "yearly": 12}

def _prepare_data(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    asset_cols = [c for c in returns.columns if c != date_col]
    missing = [c for c in asset_cols if c not in target_weights.columns]
    if missing:
        raise ValueError(f"Missing target weights for: {missing}")

    ret = returns[[date_col, *asset_cols]].dropna().set_index(date_col).sort_index()
    weights = target_weights[[date_col, *asset_cols]].dropna().set_index(date_col).sort_index()
    common_dates = ret.index.intersection(weights.index)
    return ret.loc[common_dates], weights.loc[common_dates], asset_cols

def _cost_vector(asset_cols: list[str], transaction_costs: float | dict[str, float]) -> np.ndarray:
    if isinstance(transaction_costs, dict):
        return np.array([transaction_costs.get(asset, 0.0) for asset in asset_cols], dtype=float)
    return np.repeat(float(transaction_costs), len(asset_cols))


def _is_rebalance_date(i: int, frequency: str, date) -> bool:
    month = pd.Timestamp(date).month
    return (
        frequency in ["M", "monthly"]
        or (frequency in ["Q", "quarterly"] and month % 3 == 0)
        or (frequency in ["Y", "yearly"] and month == 12)
        or ((i + 1) % FREQ_MONTHS[frequency] == 0 and pd.isna(date))
    )


def _range_weights(pre_w: np.ndarray, target_w: np.ndarray, threshold: float) -> np.ndarray:
    diff = pre_w - target_w
    max_diff = np.abs(diff).max()
    return pre_w if max_diff <= threshold else target_w + diff * threshold / max_diff


def _simulate(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    mode: str,
    frequency: str | None = None,
    threshold: float = 0.03,
    transaction_costs: float | dict[str, float] = 0.001,
    date_col: str = "Date",
) -> pd.DataFrame:
    ret, targets, asset_cols = _prepare_data(returns, target_weights, date_col)
    costs = _cost_vector(asset_cols, transaction_costs)
    wealth, weights = 1.0, targets.iloc[0].to_numpy(float)
    rows = []

    for i, (date, row) in enumerate(ret.iterrows()):
        start_wealth, start_w = wealth, weights.copy()

        # Apply asset returns before evaluating the rebalancing rule.
        values = wealth * weights * (1 + row.to_numpy(float))
        pre_wealth = values.sum()
        pre_w = values / pre_wealth
        target_w = targets.loc[date].to_numpy(float)
        post_w = pre_w.copy()

        if mode == "periodic" and _is_rebalance_date(i, frequency, date):
            post_w = target_w
        elif mode == "threshold" and _is_rebalance_date(i, frequency, date):
            post_w = target_w if np.abs(pre_w - target_w).max() > threshold else pre_w
        elif mode == "range" and _is_rebalance_date(i, frequency, date):
            post_w = _range_weights(pre_w, target_w, threshold)

        trades = np.abs(post_w - pre_w)
        cost = trades @ costs
        wealth = pre_wealth * (1 - cost)
        weights = post_w

        rows.append(
            {
                "Date": date,
                "Portfolio_Return": wealth / start_wealth - 1,
                "Portfolio_Value": wealth,
                "Turnover": trades.sum() / 2,
                "Transaction_Cost": cost,
                "Rebalanced": not np.allclose(post_w, pre_w),
                **{f"{asset}_Weight_Start": w for asset, w in zip(asset_cols, start_w)},
                **{f"{asset}_Weight_Pre_Rebal": w for asset, w in zip(asset_cols, pre_w)},
                **{f"{asset}_Weight_End": w for asset, w in zip(asset_cols, weights)},
            }
        )

    return pd.DataFrame(rows)


def buy_and_hold_multi_asset(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    return _simulate(returns, target_weights, mode="buy_and_hold", date_col=date_col)


def periodic_rebalancing_multi_asset(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    frequency: str,
    transaction_costs: float | dict[str, float] = 0.001,
    date_col: str = "Date",
) -> pd.DataFrame:
    return _simulate(returns, target_weights, "periodic", frequency, transaction_costs=transaction_costs, date_col=date_col)


def threshold_rebalancing_multi_asset(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    frequency: str,
    threshold: float = 0.03,
    transaction_costs: float | dict[str, float] = 0.001,
    date_col: str = "Date",
) -> pd.DataFrame:
    return _simulate(returns, target_weights, "threshold", frequency, threshold, transaction_costs, date_col)


def range_rebalancing_multi_asset(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    frequency: str,
    threshold: float = 0.03,
    transaction_costs: float | dict[str, float] = 0.001,
    date_col: str = "Date",
) -> pd.DataFrame:
    return _simulate(returns, target_weights, "range", frequency, threshold, transaction_costs, date_col)


def run_multi_asset_rebalancing_strategy(
    strategy: str,
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    frequency: str = "Q",
    threshold: float = 0.03,
    transaction_costs: float | dict[str, float] = 0.001,
    date_col: str = "Date",
) -> pd.DataFrame:
    funcs = {
        "buy_and_hold": buy_and_hold_multi_asset,
        "periodic": periodic_rebalancing_multi_asset,
        "threshold": threshold_rebalancing_multi_asset,
        "range": range_rebalancing_multi_asset,
    }
    kwargs = {"returns": returns, "target_weights": target_weights, "date_col": date_col}
    if strategy != "buy_and_hold":
        kwargs |= {"frequency": frequency, "transaction_costs": transaction_costs}
    if strategy in ["threshold", "range"]:
        kwargs["threshold"] = threshold
    return funcs[strategy](**kwargs)
