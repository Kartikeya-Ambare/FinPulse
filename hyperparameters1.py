# hyperparameters.py
# ─────────────────────────────────────────────────────────────────────────────
# Per-Stock Hyperparameter Configuration
#
# HOW TO ADD A NEW STOCK:
#   1. Add a new entry to STOCK_HYPERPARAMS below using the ticker as key
#   2. Set all 8 parameters (see PARAMETER GUIDE at the bottom)
#   3. Save and re-run — no other file needs to be changed
#
# If a ticker is NOT listed here, DEFAULT_PARAMS is used automatically.
# ─────────────────────────────────────────────────────────────────────────────

# STOCK_HYPERPARAMS = {

#     "RELIANCE.NS": {
#         "seq_len":        45,
#         "epochs":         120,
#         "batch_size":     32,
#         "learning_rate":  0.0008,
#         "dropout":        0.35,   # ↑ was 0.25 — more regularisation
#         "lstm_units":     128,
#         "gru_units":      128,
#         "conv_filters":   64,
#     },

#     "TCS.NS": {
#         "seq_len":        30,
#         "epochs":         100,
#         "batch_size":     32,
#         "learning_rate":  0.001,
#         "dropout":        0.30,   # ↑ was 0.20
#         "lstm_units":     64,
#         "gru_units":      64,
#         "conv_filters":   64,
#     },

#     "INFY.NS": {
#         "seq_len":        35,
#         "epochs":         110,
#         "batch_size":     16,
#         "learning_rate":  0.0009,
#         "dropout":        0.30,   # ↑ was 0.20
#         "lstm_units":     96,
#         "gru_units":      96,
#         "conv_filters":   64,
#     },

#     "HDFCBANK.NS": {
#         "seq_len":        40,
#         "epochs":         100,
#         "batch_size":     32,
#         "learning_rate":  0.001,
#         "dropout":        0.38,   # ↑ was 0.30
#         "lstm_units":     64,
#         "gru_units":      128,
#         "conv_filters":   128,
#     },

#     "WIPRO.NS": {
#         "seq_len":        30,
#         "epochs":         90,
#         "batch_size":     32,
#         "learning_rate":  0.001,
#         "dropout":        0.30,   # ↑ was 0.20
#         "lstm_units":     64,
#         "gru_units":      64,
#         "conv_filters":   64,
#     },

#     "ICICIBANK.NS": {
#         "seq_len":        40,
#         "epochs":         110,
#         "batch_size":     32,
#         "learning_rate":  0.0008,
#         "dropout":        0.35,   # ↑ was 0.25
#         "lstm_units":     96,
#         "gru_units":      96,
#         "conv_filters":   64,
#     },

#     "BHARTIARTL.NS": {
#         "seq_len":        35,
#         "epochs":         100,
#         "batch_size":     32,
#         "learning_rate":  0.001,
#         "dropout":        0.30,   # ↑ was 0.20
#         "lstm_units":     64,
#         "gru_units":      64,
#         "conv_filters":   64,
#     },

#     "ITC.NS": {
#         "seq_len":        30,
#         "epochs":         90,
#         "batch_size":     64,
#         "learning_rate":  0.001,
#         "dropout":        0.25,   # ↑ was 0.15
#         "lstm_units":     64,
#         "gru_units":      64,
#         "conv_filters":   32,
#     },

#     "SBIN.NS": {
#         "seq_len":        40,
#         "epochs":         100,
#         "batch_size":     32,
#         "learning_rate":  0.001,
#         "dropout":        0.35,   # ↑ was 0.25
#         "lstm_units":     96,
#         "gru_units":      96,
#         "conv_filters":   64,
#     },

#     "ADANIENT.NS": {
#         "seq_len":        50,
#         "epochs":         130,
#         "batch_size":     16,
#         "learning_rate":  0.0006,
#         "dropout":        0.40,   # ↑ was 0.35
#         "lstm_units":     128,
#         "gru_units":      128,
#         "conv_filters":   128,
#     },

#     "HINDUNILVR.NS": {
#         "seq_len":        30,
#         "epochs":         90,
#         "batch_size":     64,
#         "learning_rate":  0.001,
#         "dropout":        0.25,   # ↑ was 0.15
#         "lstm_units":     64,
#         "gru_units":      64,
#         "conv_filters":   32,
#     },

#     "BAJFINANCE.NS": {
#         "seq_len":        45,
#         "epochs":         120,
#         "batch_size":     32,
#         "learning_rate":  0.0008,
#         "dropout":        0.38,   # ↑ was 0.30
#         "lstm_units":     128,
#         "gru_units":      128,
#         "conv_filters":   64,
#     },
STOCK_HYPERPARAMS = {
 
    "RELIANCE.NS": {
        "seq_len":        42,     
        "epochs":         110,     
        "batch_size":     32,
        "learning_rate":  0.00085,
        "dropout":        0.32,    
        "lstm_units":     112,     
        "gru_units":      112,     
        "conv_filters":   60,     
    },
 
    "TCS.NS": {
        "seq_len":        32,      # ↑ was 30 — more context
        "epochs":         95,      # ↓ was 100 — avoid overtraining
        "batch_size":     32,
        "learning_rate":  0.00095, # ↓ was 0.001 — finer adjustments
        "dropout":        0.28,    # ↓ was 0.30 — less regularization
        "lstm_units":     60,      # ↓ was 64 — lighter model
        "gru_units":      60,      # ↓ was 64 — lighter model
        "conv_filters":   60,      # ↓ was 64 — balanced filters
    },
 
    "INFY.NS": {
        "seq_len":        33,      # ↓ was 35 — reduce overfitting
        "epochs":         105,     # ↓ was 110 — earlier convergence
        "batch_size":     16,
        "learning_rate":  0.00095, # ↓ was 0.0009 — fine-tune
        "dropout":        0.29,    # ↓ was 0.30 — slight reduction
        "lstm_units":     92,      # ↓ was 96 — balance capacity
        "gru_units":      92,      # ↓ was 96 — balance capacity
        "conv_filters":   62,      # ↑ was 64 — maintain adequate feature extraction
    },
 
    "HDFCBANK.NS": {
        "seq_len":        38,      # ↓ was 40 — reduce temporal complexity
        "epochs":         105,     # ↑ was 100 — allow better training
        "batch_size":     32,
        "learning_rate":  0.00095, # ↓ was 0.001 — finer tuning
        "dropout":        0.36,    # ↓ was 0.38 — reduce over-regularization
        "lstm_units":     62,      # ↓ was 64 — lighter model
        "gru_units":      120,     # ↓ was 128 — reduce model size
        "conv_filters":   120,     # ↓ was 128 — balanced complexity
    },
 
    "WIPRO.NS": {
        "seq_len":        32,      # ↑ was 30 — more lookback
        "epochs":         88,      # ↓ was 90 — avoid overtraining
        "batch_size":     32,
        "learning_rate":  0.00098, # ↓ was 0.001 — fine-tune
        "dropout":        0.29,    # ↓ was 0.30 — slight reduction
        "lstm_units":     62,      # ↓ was 64 — balance capacity
        "gru_units":      62,      # ↓ was 64 — balance capacity
        "conv_filters":   60,      # ↓ was 64 — reduced filters
    },
 
    "ICICIBANK.NS": {
        "seq_len":        38,      # ↓ was 40 — reduce overfitting
        "epochs":         108,     # ↓ was 110 — earlier convergence
        "batch_size":     32,
        "learning_rate":  0.00082, # ↓ was 0.0008 — finer tuning
        "dropout":        0.33,    # ↓ was 0.35 — reduce over-regularization
        "lstm_units":     92,      # ↓ was 96 — balanced model
        "gru_units":      92,      # ↓ was 96 — balanced model
        "conv_filters":   62,      # ↓ was 64 — reduce filters
    },
 
    "BHARTIARTL.NS": {
        "seq_len":        33,      # ↓ was 35 — reduce complexity
        "epochs":         98,      # ↓ was 100 — earlier convergence
        "batch_size":     32,
        "learning_rate":  0.00098, # ↓ was 0.001 — fine-tune
        "dropout":        0.29,    # ↓ was 0.30 — slight reduction
        "lstm_units":     62,      # ↓ was 64 — lighter model
        "gru_units":      62,      # ↓ was 64 — lighter model
        "conv_filters":   60,      # ↓ was 64 — reduced filters
    },
 
    "ITC.NS": {
        "seq_len":        32,      # ↑ was 30 — more context
        "epochs":         88,      # ↓ was 90 — avoid overtraining
        "batch_size":     60,      # ↓ was 64 — better gradient noise
        "learning_rate":  0.00098, # ↓ was 0.001 — fine-tune
        "dropout":        0.24,    # ↓ was 0.25 — slight reduction
        "lstm_units":     62,      # ↓ was 64 — lighter model
        "gru_units":      62,      # ↓ was 64 — lighter model
        "conv_filters":   30,      # ↓ was 32 — minimal filters
    },
 
    "SBIN.NS": {
        "seq_len":        38,      # ↓ was 40 — reduce overfitting
        "epochs":         98,      # ↓ was 100 — earlier convergence
        "batch_size":     32,
        "learning_rate":  0.00098, # ↓ was 0.001 — fine-tune
        "dropout":        0.33,    # ↓ was 0.35 — reduce over-regularization
        "lstm_units":     92,      # ↓ was 96 — balanced model
        "gru_units":      92,      # ↓ was 96 — balanced model
        "conv_filters":   62,      # ↓ was 64 — reduce filters
    },
 
    "ADANIENT.NS": {
        "seq_len":        48,      # ↓ was 50 — reduce complexity
        "epochs":         125,     # ↓ was 130 — control training length
        "batch_size":     16,
        "learning_rate":  0.00062, # ↑ was 0.0006 — smoother updates
        "dropout":        0.38,    # ↓ was 0.40 — reduce over-regularization
        "lstm_units":     120,     # ↓ was 128 — lighter model
        "gru_units":      120,     # ↓ was 128 — lighter model
        "conv_filters":   120,     # ↓ was 128 — balanced filters
    },
 
    "HINDUNILVR.NS": {
        "seq_len":        32,      # ↑ was 30 — more context
        "epochs":         88,      # ↓ was 90 — avoid overtraining
        "batch_size":     60,      # ↓ was 64 — better gradient noise
        "learning_rate":  0.00098, # ↓ was 0.001 — fine-tune
        "dropout":        0.24,    # ↓ was 0.25 — slight reduction
        "lstm_units":     62,      # ↓ was 64 — lighter model
        "gru_units":      62,      # ↓ was 64 — lighter model
        "conv_filters":   30,      # ↓ was 32 — minimal filters
    },
 
    "BAJFINANCE.NS": {
        "seq_len":        42,      # ↓ was 45 — reduce overfitting
        "epochs":         115,     # ↓ was 120 — control training
        "batch_size":     32,
        "learning_rate":  0.00082, # ↓ was 0.0008 — fine-tune
        "dropout":        0.36,    # ↓ was 0.38 — reduce over-regularization
        "lstm_units":     120,     # ↓ was 128 — lighter model
        "gru_units":      120,     # ↓ was 128 — lighter model
        "conv_filters":   62,      # ↓ was 64 — reduce filters
    },
    # ── ADD NEW STOCKS BELOW THIS LINE ────────────────────────────────────────
    # Copy-paste this template and fill in values:
    #
    # "TICKER.NS": {
    #     "seq_len":       30,
    #     "epochs":        100,
    #     "batch_size":    32,
    #     "learning_rate": 0.001,
    #     "dropout":       0.20,
    #     "lstm_units":    64,
    #     "gru_units":     64,
    #     "conv_filters":  64,
    # },
}

# ─────────────────────────────────────────────────────────────────────────────
# Default fallback — used when ticker is NOT in STOCK_HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "seq_len":        30,
    "epochs":         100,
    "batch_size":     32,
    "learning_rate":  0.001,
    "dropout":        0.30,   # ↑ was 0.20 — more regularisation
    "lstm_units":     64,
    "gru_units":      64,
    "conv_filters":   64,
}


def get_hyperparams(ticker: str) -> dict:
    """
    Returns hyperparameters for the given ticker.
    Falls back to DEFAULT_PARAMS if ticker is not configured.
    """
    if ticker in STOCK_HYPERPARAMS:
        return STOCK_HYPERPARAMS[ticker].copy()
    print(f"[hyperparams] No config found for {ticker} — using DEFAULT_PARAMS")
    return DEFAULT_PARAMS.copy()


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GUIDE
# ─────────────────────────────────────────────────────────────────────────────
# seq_len        : Lookback window in days. Higher = more context but slower.
#                  Recommended range: 20–60
#
# epochs         : Max training iterations. EarlyStopping stops early if
#                  val_loss plateaus. Recommended range: 80–150
#
# batch_size     : Samples per gradient update. Smaller = noisier gradients
#                  but can generalise better. Options: 16, 32, 64
#
# learning_rate  : Adam optimizer step size.
#                  Recommended range: 0.0005–0.002
#
# dropout        : Fraction of units randomly dropped after recurrent blocks.
#                  Higher for noisy/volatile stocks. Range: 0.10–0.40
#
# lstm_units     : Units in the LSTM layer. Options: 32, 64, 96, 128
#
# gru_units      : Units in each BiGRU layer (doubled by Bidirectional wrapper).
#                  Options: 32, 64, 96, 128
#
# conv_filters   : Filters in Conv1D layers (second layer uses 2×).
#                  Options: 32, 64, 128
# ─────────────────────────────────────────────────────────────────────────────
