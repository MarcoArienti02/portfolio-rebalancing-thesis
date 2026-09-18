import pandas as pd
from src.data.data_master import monthly_rf_return, simple_returns


EQUITY_USD = ["US_Equity_USD", "UK_Equity_USD", "DE_Equity_USD"]
ALT_USD = ["Gold_USD", "Real_Estate_USD", "Commodities_USD"]


def build_multi_asset_usd_levels(levels: pd.DataFrame) -> pd.DataFrame:
    out = levels[["Date"]].copy()

    # Convert UK equity to USD; US and German equity series are already in USD.
    out["US_Equity_USD_TR"] = levels["USA_Equity_TR"]
    out["UK_Equity_USD_TR"] = levels["UK_Equity_TR"] * levels["GBPUSD"]
    out["DE_Equity_USD_TR"] = levels["GER_Equity_USD_TR"]

    # UK and German synthetic bond indices are local-currency series converted to USD.
    for maturity in (5, 10):
        out[f"US_Bond_{maturity}Y_USD_TR"] = levels[f"US_Bond_{maturity}Y_TR_Index"]
        out[f"UK_Bond_{maturity}Y_USD_TR"] = levels[f"UK_Bond_{maturity}Y_TR_Index"] * levels["GBPUSD"]
        out[f"DE_Bond_{maturity}Y_USD_TR"] = levels[f"DE_Bond_{maturity}Y_TR_Index"] * levels["GER_USD_per_Local"]

    # Alternative assets are already denominated in USD in the licensed source data.
    out["Gold_USD"] = levels["Gold"]
    out["Real_Estate_USD_TR"] = levels["Real_Estate_TR"]
    out["Commodities_USD_TR"] = levels["Commodities_TR"]

    return out


def build_multi_asset_usd_returns(levels: pd.DataFrame) -> pd.DataFrame:
    usd = build_multi_asset_usd_levels(levels)
    out = usd[["Date"]].copy()

    # Returns are calculated after currency conversion.
    for col in usd.columns.drop("Date"):
        out[f"{col.removesuffix('_TR')}_Return"] = simple_returns(usd[col])

    # Use the US T-Bill as the USD cash proxy.
    out["Cash_USD_Return"] = monthly_rf_return(levels["US_3M_RF"])
    return out


def select_multi_asset_universe(
    returns: pd.DataFrame,
    bond_maturity: int = 10,
    include_cash: bool = False,
) -> pd.DataFrame:
    bond_cols = [f"{country}_Bond_{bond_maturity}Y_USD_Return" for country in ("US", "UK", "DE")]
    cols = [f"{col}_Return" for col in EQUITY_USD] + bond_cols + [f"{col}_Return" for col in ALT_USD]

    # Cash can be an investable asset or only the risk-free series for performance metrics.
    if include_cash:
        cols.append("Cash_USD_Return")

    return returns[["Date", *cols]].dropna().reset_index(drop=True)
