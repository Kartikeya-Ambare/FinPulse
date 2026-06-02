# 📈 FinPulse Quant Engine

> **Multi-Stock Portfolio Prediction · GARCH Volatility · Monte Carlo Risk Analytics**
> A fully self-contained Streamlit dashboard — no backend server required.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Overview

FinPulse Quant Engine is an end-to-end ML-powered stock analysis platform for **NSE (National Stock Exchange of India)** equities. It fetches 3 years of live OHLCV data, trains a hybrid **Conv1D + LSTM + GRU** deep learning model per stock, computes GARCH volatility, runs Monte Carlo simulations, and presents everything in a rich interactive Streamlit dashboard — all in a single command.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Deep Learning Prediction** | Per-stock Conv1D → LSTM → GRU model with tuned hyperparameters; predicts OHLC for Day+1 and Day+2 |
| **GARCH Volatility** | GARCH(1,1) model with Risk Adjustment Factor (RAF) for annualised volatility estimates |
| **Monte Carlo Simulation** | 2,000 portfolio paths over 30 trading days |
| **3 VaR Methods** | Historical Simulation · Parametric (Variance-Covariance) · Monte Carlo VaR |
| **Backtesting Engine** | BUY/SELL signal replay on the validation set with cumulative P&L and direction accuracy |
| **Financial Health KPIs** | 20+ KPIs across Income Statement, Balance Sheet, Cash Flow, and Valuation with colour-coded health badges |
| **Candlestick Charts** | Last 90 days of OHLCV with volume bars |
| **Correlation Heatmap** | Log-return correlation matrix across selected stocks |
| **Portfolio Allocation** | Pie chart, bar chart, weight breakdown table, and portfolio trend |

---

## 🏗️ Architecture

```
finpulse/
├── app.py                   # ← Streamlit dashboard + embedded ML pipeline (single entry point)
├── data_loader.py           # yfinance data fetching + fundamental data
├── feature_engineering.py  # Technical indicator construction & sequence creation
├── ml_model.py              # Conv1D + LSTM + GRU PyTorch model, training & inference
├── volatility_model.py      # GARCH(1,1) volatility estimation
├── monte_carlo_simulation.py# Portfolio path simulation
├── portfolio_prediction.py  # Stock & portfolio summary computation
├── risk_model.py            # VaR, CVaR, Sharpe ratio, max drawdown
├── hyperparameters.py       # Per-ticker hyperparameter registry
├── visualization.py         # All Plotly chart functions
├── requirements.txt
└── README.md
```

> **Note:** `api_server.py` (the old FastAPI backend) is no longer needed. All pipeline logic is embedded directly in `app.py` via the `run_analysis()` function.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/finpulse-quant-engine.git
cd finpulse-quant-engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU Support (optional):** For faster training with CUDA 12.1:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Select Stocks** — choose 1–11 NSE tickers from the multiselect (e.g., `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`)
2. **Set Quantities** — enter the number of shares held per stock
3. **Run Analysis** — click **🚀 Run Full Analysis** and wait ~2–5 minutes per stock for model training
4. **Explore Tabs** — navigate through the 6 dashboard tabs below

### Dashboard Tabs

| Tab | Content |
|---|---|
| 📈 **Stock Predictions** | Predicted vs. actual close prices, Day+1 and Day+2 OHLC forecast, per-stock metrics |
| 🕯️ **Candlestick Charts** | 90-day historical OHLCV candlestick with volume |
| 💼 **Portfolio Panel** | Portfolio value trend, allocation pie, quantity bar chart, weight breakdown |
| ⚠️ **Risk Analytics** | 3×VaR methods, CVaR, Sharpe Ratio, Max Drawdown, GARCH table, correlation heatmap |
| 🔬 **Backtest & Accuracy** | Consolidated accuracy table (R², MAE, RMSE, MAPE, Direction Acc), per-stock backtest charts |
| 🏥 **Financial Health KPIs** | 20+ KPIs across Income Statement / Balance Sheet / Cash Flow / Valuation |

---

## 📦 Supported Tickers (pre-configured)

| Ticker | Company |
|---|---|
| `RELIANCE.NS` | Reliance Industries |
| `TCS.NS` | Tata Consultancy Services |
| `INFY.NS` | Infosys |
| `HDFCBANK.NS` | HDFC Bank |
| `WIPRO.NS` | Wipro |
| `ICICIBANK.NS` | ICICI Bank |
| `ITC.NS` | ITC Ltd |
| `SBIN.NS` | State Bank of India |
| `ADANIENT.NS` | Adani Enterprises |
| `HINDUNILVR.NS` | Hindustan Unilever |
| `BAJFINANCE.NS` | Bajaj Finance |

To add a new ticker, add its hyperparameter config to `hyperparameters.py` (see the file's guide at the top). Any valid `yfinance` NSE symbol works even without a custom config — `DEFAULT_PARAMS` will be used automatically.

---

## ⚙️ Model Details

### Deep Learning Architecture

```
Input Sequences (seq_len × n_features)
        ↓
Conv1D  (conv_filters kernels, kernel_size=3, ReLU)
        ↓
LSTM    (lstm_units, return_sequences=True)
        ↓
GRU     (gru_units)
        ↓
Dropout (per-stock rate)
        ↓
Linear  → 4 outputs (ΔOpen, ΔHigh, ΔLow, ΔClose returns)
        ↓
Reconstruct OHLC prices from predicted returns
```

### Volatility Model

- **GARCH(1,1)** fitted on log-returns
- Annualised by multiplying daily vol by √252
- Adjusted by per-stock **Risk Adjustment Factor (RAF)** from fundamental data

### Risk Metrics

| Metric | Method |
|---|---|
| Historical VaR 95% / 99% | Empirical percentile of daily portfolio returns |
| CVaR 95% | Mean of returns below VaR threshold |
| Parametric VaR 95% / 99% | Normal distribution assumption (μ - z·σ) |
| Monte Carlo VaR | 5th percentile of simulated 30-day portfolio values |
| Sharpe Ratio | Annualised (μ - r_f) / σ, r_f = 6% |
| Max Drawdown | Peak-to-trough decline in portfolio value |

---

## 🔧 Configuration

All per-stock model hyperparameters live in `hyperparameters.py`:

```python
# Example entry
"TCS.NS": {
    "seq_len":       28,      # lookback window (days)
    "epochs":        80,      # training epochs
    "batch_size":    16,      # mini-batch size
    "learning_rate": 0.0012,  # Adam LR
    "dropout":       0.22,    # dropout rate
    "lstm_units":    112,     # LSTM hidden size
    "gru_units":     112,     # GRU hidden size
    "conv_filters":  56,      # Conv1D filter count
},
```

Global simulation settings in `app.py`:

```python
MC_DAYS       = 30    # Monte Carlo horizon (trading days)
N_SIMULATIONS = 2000  # Number of simulated paths
```

---

## 📊 Accuracy Metrics

| Metric | Description |
|---|---|
| **R²** | Coefficient of determination (capped at ~0.91–0.92) |
| **MAE** | Mean Absolute Error in ₹ |
| **RMSE** | Root Mean Squared Error in ₹ |
| **MAPE** | Mean Absolute Percentage Error |
| **Direction Accuracy** | % of days the model correctly predicted up/down movement |

---

## 🛠️ Requirements

- Python **3.10+**
- Internet connection (for live yfinance data)
- RAM: **≥ 8 GB** recommended (16 GB for training multiple stocks simultaneously)
- GPU: optional but significantly speeds up training

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

> FinPulse Quant Engine is for **educational and research purposes only**.
> It does **not** constitute financial advice. Past model performance does not guarantee future results.
> Always consult a qualified financial advisor before making investment decisions.

---

<div align="center">
  Built with ❤️ using Streamlit · PyTorch · Plotly · yfinance
</div>
