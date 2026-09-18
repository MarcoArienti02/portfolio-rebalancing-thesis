# Portfolio Rebalancing Strategies and Dynamic Asset Allocation

This repository presents the Python implementation developed for an MSc thesis on stock-bond portfolio rebalancing and dynamic asset allocation. The research replicates and extends the framework of Dichtl, Drobetz, and Wambach (2014), with explicit treatment of transaction costs, turnover, statistical uncertainty, and rolling portfolio optimization.

## Research objective

The project asks whether disciplined rebalancing improves the risk-adjusted performance of diversified portfolios after trading costs. The empirical design compares US, UK, and German stock-bond portfolios, extends the original sample through 2025, and studies dynamic multi-asset allocations based on Global Minimum Variance (GMV) and Risk Parity.

## Main research components

- Buy-and-hold, monthly, quarterly, and yearly rebalancing
- Threshold and range-based rebalancing rules
- Proportional transaction-cost and portfolio-turnover modelling
- US, UK, and German equity/government-bond portfolios
- Sharpe, Sortino, Omega, drawdown, return, and volatility analysis
- Rolling covariance estimation, GMV, and Risk Parity portfolios
- Rolling-window evaluation and stationary-bootstrap inference
- Jobson-Korkie tests with Memmel correction for Sharpe-ratio differences
- Construction and external validation of synthetic government-bond total-return series

## Python implementation

The codebase separates the core research logic from the private data-preparation workflow. It includes reusable functions for:

- aligning financial time series and calculating returns;
- approximating constant-maturity bond total returns from yields and duration;
- simulating two-asset and multi-asset rebalancing strategies;
- applying asset-specific transaction costs and measuring turnover;
- computing risk-adjusted performance and drawdown statistics;
- estimating rolling GMV and Risk Parity weights without look-ahead bias; and
- generating bootstrap confidence intervals and parametric robustness tests.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_bond_total_return_validation.ipynb
│   ├── 02_rolling_window_rebalancing.ipynb
│   ├── 03_stationary_bootstrap.ipynb
│   ├── 04_dynamic_allocation_gmv_risk_parity.ipynb
│   └── 05_sharpe_ratio_significance_tests.ipynb
└── src/
    ├── bootstrap/
    ├── data/
    ├── metrics/
    ├── optimization/
    └── portfolio/
```

The notebooks are a curated subset of the thesis workflow. Their saved outputs have been removed to avoid redistributing licensed-data-derived tables or figures.

## Notebook guide

| Notebook | Focus |
|---|---|
| `01_bond_total_return_validation.ipynb` | Duration-based government-bond total-return construction and benchmark validation |
| `02_rolling_window_rebalancing.ipynb` | Five- and ten-year rolling evaluation of 60/40 rebalancing rules |
| `03_stationary_bootstrap.ipynb` | Politis-Romano stationary bootstrap and confidence intervals |
| `04_dynamic_allocation_gmv_risk_parity.ipynb` | Rolling GMV/Risk Parity weights and multi-asset rebalancing backtests |
| `05_sharpe_ratio_significance_tests.ipynb` | Jobson-Korkie/Memmel robustness tests for Sharpe-ratio differences |

## Data

The thesis uses Bloomberg and other licensed financial data. Those raw and processed datasets are **not included and may not be redistributed**. The public code therefore documents the required schemas in [`data/README.md`](data/README.md), while all notebook outputs are cleared.

The repository does not fabricate substitute thesis results. Users with appropriate data access can provide compatible files under `data/processed/` and run the notebooks locally.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run notebooks from the repository root or from the `notebooks/` directory.

## Academic context

- MSc in Economics and Finance
- University of Milano-Bicocca
- Expected graduation: November 2026

## Status

The thesis is ongoing. This repository is a curated public-facing code portfolio and not a complete copy of the private research workspace.

## Disclaimer

This repository is provided for academic and educational purposes only. It does not constitute investment advice, a recommendation, or an offer to buy or sell financial instruments.
