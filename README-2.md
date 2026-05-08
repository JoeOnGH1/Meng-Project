# Reinforcement Learning Framework for Intraday Electricity Market Bidding

MEng Electronic and Electrical Engineering — University of Bath — 2025/26

This repository contains the full simulation and training pipeline for a reinforcement learning framework comparing Proximal Policy Optimisation (PPO) and Deep Deterministic Policy Gradient (DDPG) for intraday electricity market bidding in a simulated German DE-LU continuous intraday market.

---

## Repository Contents

| Notebook | Description |
|---|---|
| `Data_Extraction.ipynb` | Downloads and preprocesses day-ahead electricity prices from ENTSO-E (document types A44, A69) and wind generation and system load data from SMARD (filter codes 4067, 1225, 410). Computes Wind Forecast Error and saves cleaned datasets. |
| `Simulation_GARCH_X_volatility.ipynb` | Fits a GARCH(1,1) volatility model on training data, constructs the OU spread simulation environment, and generates the primary GARCH simulation environment used in all walk-forward experiments. |
| `Simulation_Heston_+_WFE.ipynb` | Documents the attempted Heston stochastic volatility calibration, including the boundary convergence failure, and the subsequent pivot to GARCH. |
| `Full_state_CV_and_WF_validation.ipynb` | Runs nested cross-validation hyperparameter tuning and five-fold walk-forward validation for PPO and DDPG in the full state configuration (nine state features including spread_norm). |
| `Reduced_state_CV_and_WF_validation.ipynb` | Runs the reduced state ablation across four folds, removing spread_norm and price_da_norm to test whether agents can discover profitable strategies without direct access to the mean-reverting signal. |
| `PnL_distributions.ipynb` | Generates episode PnL distribution plots and action distribution analysis across all folds and state configurations. |

---

## Running Order

1. `Data_Extraction.ipynb`
2. `Simulation_GARCH_X_volatility.ipynb`
3. `Simulation_Heston_+_WFE.ipynb` (optional, documents failed calibration)
4. `Full_state_CV_and_WF_validation.ipynb`
5. `Reduced_state_CV_and_WF_validation.ipynb`
6. `PnL_distributions.ipynb`

---

## Data Sources

All data is downloaded programmatically from publicly available sources:

- **ENTSO-E Transparency Platform** — day-ahead prices (document type A44, EIC area 10Y1001A1001A82H) and wind power forecasts (document type A69, series B18 and B19)
- **SMARD** (Bundesnetzagentur) — wind generation (filter code 4067), system load (filter code 1225), and solar generation (filter code 410)

No raw data files are included in this repository. All data can be re-downloaded by running `Data_Extraction.ipynb`.

Model checkpoints are not included due to file size.

---

## Environment and Dependencies

All notebooks were developed and run in Google Colab using a T4 GPU. Key dependencies include:

- Python 3.10
- stable-baselines3
- gymnasium
- arch (for GARCH fitting)
- pandas, numpy, matplotlib
- entsoe-py

---

## Important: File Path Configuration

> **Disclaimer:** All notebooks use Google Drive mount paths (e.g. `/content/drive/MyDrive/MEng Project/...`) that are specific to the original development environment. If you wish to run these notebooks, you will need to update all file paths to match your own environment, whether that is a local directory structure or your own Google Drive layout. A find-and-replace on the path prefix in each notebook is the quickest way to do this. Aditionally, API keys have been redacted, users will need to request ENTSO-E API keys in order to extract data. Do so by politely emailing transparency@entsoe.eu with an API access request.

---

## Project Context

This project was submitted in partial fulfilment of the requirements for the MEng Electronic and Electrical Engineering degree at the University of Bath. The central finding is that DDPG fails to learn a profitable response within a simulated intraday market where a deterministic three-line rule and a trained PPO agent both trade profitably, suggesting a structural mismatch between continuous off-policy methods and episodic categorical trading tasks.
