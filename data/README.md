# Data requirements

This public repository does not contain Bloomberg data, licensed benchmarks, thesis datasets, or derived research outputs. Place compatible input files in `data/processed/` only if you have the legal right to use them. That directory is excluded from Git.

## Country portfolio files

The rolling-window, bootstrap, and Sharpe-test notebooks expect:

- `us_data.csv`
- `uk_data.csv`
- `de_data.csv`

Minimum schema:

| Column | Description |
|---|---|
| `Date` | Month-end date |
| `Equity_Return` | Monthly total return of the country equity index, decimal |
| `Bond_10Y_Return` | Monthly 10-year government-bond total return, decimal |
| `RF_Return` | Monthly risk-free return, decimal |

## Bond-validation files

`01_bond_total_return_validation.ipynb` expects cleaned yield series and licensed benchmark returns with month-end dates. The required filenames and country-specific column selections are declared near the top of the notebook.

## Multi-asset file

`04_dynamic_allocation_gmv_risk_parity.ipynb` expects `multi_asset_usd_returns.csv` with a `Date` column and monthly USD returns for:

- US, UK, and German equities;
- US, UK, and German 10-year government bonds;
- gold, real estate, and commodities; and
- a USD cash/risk-free proxy.

The exact programmatic column names are defined in `src/data/multi_asset_usd.py`.

## Conventions

- Returns are decimals, not percentages.
- Dates should be unique month-end observations in ascending order.
- No missing values should remain in the final estimation sample.
- Currency conversion must be completed before multi-asset returns are calculated.
- Do not commit licensed or proprietary files to this repository.

