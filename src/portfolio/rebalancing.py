import numpy as np
import pandas as pd

FREQ_MONTHS = {"M": 1, "Q": 3, "Y": 12, "monthly": 1, "quarterly": 3, "yearly": 12}

def _prepare_returns(stock_ret: pd.Series, bond_ret: pd.Series, dates: pd.Series | None = None) -> pd.DataFrame:
    data = pd.concat({"Stock_Return": stock_ret, "Bond_Return": bond_ret}, axis=1)
    if dates is not None:
        data["Date"] = pd.to_datetime(dates)
    return data.dropna(subset=["Stock_Return", "Bond_Return"]).reset_index(drop=True)

# Return whether the portfolio should be rebalanced on the current date.
def _is_rebalance_date(i: int, frequency: str, date=None) -> bool:
    if date is None or pd.isna(date):
        return (i + 1) % FREQ_MONTHS[frequency] == 0
    # Fall back to the period number when dates are unavailable.
    month = pd.Timestamp(date).month
    return (
        frequency in ["M", "monthly"]
        or (frequency in ["Q", "quarterly"] and month % 3 == 0)
        or (frequency in ["Y", "yearly"] and month == 12)
    )

# Determine the post-trade equity weight after a threshold signal.
def _target_after_signal(pre_stock_w: float, target_stock_w: float, threshold: float, mode: str) -> float | None:
    lower, upper = target_stock_w - threshold, target_stock_w + threshold
    if lower <= pre_stock_w <= upper:
        return None
    return target_stock_w if mode == "threshold" else lower if pre_stock_w < lower else upper
# Threshold rebalancing returns to target; range rebalancing returns to the nearest boundary.

# Calculate proportional transaction costs for the two-asset trade.
def _trade_cost(pre_stock_w: float, post_stock_w: float, stock_cost: float, bond_cost: float) -> float:
    trade_stock = abs(post_stock_w - pre_stock_w)
    return trade_stock * stock_cost + trade_stock * bond_cost
# Selling an equity share implies purchasing the same portfolio share in bonds.


def _simulate(
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    frequency: str | None,  # None for buy-and-hold
    mode: str,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.DataFrame:
    data = _prepare_returns(stock_ret, bond_ret, dates)
    stock_w, wealth = target_stock_w, 1.0
    rows = []

    for i, row in data.iterrows():
        start_stock_w = stock_w
        start_wealth = wealth
        stock_val = wealth * stock_w * (1 + row["Stock_Return"])
        bond_val = wealth * (1 - stock_w) * (1 + row["Bond_Return"])
        # Portfolio value and weights before rebalancing.
        pre_wealth = stock_val + bond_val
        pre_stock_w = stock_val / pre_wealth
        post_stock_w = pre_stock_w

        date = row.get("Date", None)

        # Apply the selected rebalancing rule.
        if mode == "periodic" and _is_rebalance_date(i, frequency, date):
            post_stock_w = target_stock_w
        elif mode in ["threshold", "range"] and _is_rebalance_date(i, frequency, date):
            signal = _target_after_signal(pre_stock_w, target_stock_w, threshold, mode)
            post_stock_w = pre_stock_w if signal is None else signal

        # Costs apply only to the portfolio share actually traded.
        cost = _trade_cost(pre_stock_w, post_stock_w, stock_cost, bond_cost)
        wealth = pre_wealth * (1 - cost)
        stock_w = post_stock_w

        rows.append(
            {
                "Date": date if date is not None else i,
                "Portfolio_Return": wealth / start_wealth - 1,
                "Portfolio_Value": wealth,
                "Stock_Weight_Start": start_stock_w,
                "Stock_Weight_Pre_Rebal": pre_stock_w,
                "Stock_Weight_End": stock_w,
                "Turnover": abs(post_stock_w - pre_stock_w),
                "Transaction_Cost": cost,
                "Rebalanced": not np.isclose(post_stock_w, pre_stock_w),
            }
        )

    return pd.DataFrame(rows)


def buy_and_hold(
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
) -> pd.DataFrame:
    return _simulate(stock_ret, bond_ret, None, "buy_and_hold", dates, target_stock_w=target_stock_w)


def periodic_rebalancing(
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
    frequency: str = "Q",
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.DataFrame:
    return _simulate(stock_ret, bond_ret, frequency, "periodic", dates, target_stock_w, stock_cost=stock_cost, bond_cost=bond_cost)


def threshold_rebalancing(
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
    frequency: str = "Q",
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.DataFrame:
    return _simulate(stock_ret, bond_ret, frequency, "threshold", dates, target_stock_w, threshold, stock_cost, bond_cost)


def range_rebalancing(
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
    frequency: str = "Q",
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.DataFrame:
    return _simulate(stock_ret, bond_ret, frequency, "range", dates, target_stock_w, threshold, stock_cost, bond_cost)

# Dispatch a strategy name to the corresponding simulation.
def run_rebalancing_strategy(
    strategy: str,
    stock_ret: pd.Series,
    bond_ret: pd.Series,
    dates: pd.Series | None = None,
    target_stock_w: float = 0.60,
    frequency: str = "Q",
    threshold: float = 0.03,
    stock_cost: float = 0.001,
    bond_cost: float = 0.0005,
) -> pd.DataFrame:
    funcs = {
        "buy_and_hold": buy_and_hold,
        "periodic": periodic_rebalancing,
        "threshold": threshold_rebalancing,
        "range": range_rebalancing,
    }
    # Arguments shared by all strategies.
    kwargs = {"stock_ret": stock_ret, "bond_ret": bond_ret, "dates": dates, "target_stock_w": target_stock_w}
    if strategy != "buy_and_hold":
        kwargs |= {"frequency": frequency, "stock_cost": stock_cost, "bond_cost": bond_cost}
    if strategy in ["threshold", "range"]:
        kwargs["threshold"] = threshold
    return funcs[strategy](**kwargs)
