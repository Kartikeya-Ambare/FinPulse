import numpy as np
import random
import torch
from sklearn.metrics import r2_score
from ml_model import train_model, DEVICE
from feature_engineering import build_features, create_sequences


class GeneticOptimizer:
    def __init__(self, price_data, fundamentals):
        self.price_data = price_data
        self.fundamentals = fundamentals

        # ── Hyperparameter search space ────────────────────────────────────
        self.param_grid = {
            "seq_len":       [20, 25, 30, 35, 40, 45, 50, 55, 60],
            "epochs":        [80, 90, 100, 110, 120, 130, 140, 150],
            "batch_size":    [16, 32, 64],
            "learning_rate": [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002],
            "dropout":       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "lstm_units":    [32, 64, 96, 128],
            "gru_units":     [32, 64, 96, 128],
            "conv_filters":  [32, 64, 128],
        }

    # ── Chromosome factory ─────────────────────────────────────────────────
    def _create_individual(self):
        return {k: random.choice(v) for k, v in self.param_grid.items()}

    # ── Data preparation (seq_len is genome-dependent) ─────────────────────
    def _prepare_data(self, seq_len):
        raf = self.fundamentals.get("RAF", 1.0)
        df_feat = build_features(self.price_data, raf=raf)
        X, y, scaler_X, scaler_y, feat_cols, raw_ohlc = create_sequences(df_feat, seq_len=seq_len)
        split = int(len(X) * 0.8)
        train_data = (X[:split], y[:split])
        val_data   = (X[split:], y[split:])
        n_features = X.shape[2]
        return train_data, val_data, n_features

    # ── Fitness function ───────────────────────────────────────────────────
    def _fitness(self, params):
        """
        Returns (fitness, r2).

        Fix summary:
          - seq_len : extracted → passed explicitly → stripped from **dict
                      (train_model requires it as a named arg)
          - epochs  : lives inside params → flows through **train_params
                      (no explicit epochs= to avoid duplicate kwarg error)
        """
        try:
            seq_len = params["seq_len"]
            train_data, val_data, n_features = self._prepare_data(seq_len)

            # Strip seq_len so it is not passed twice (explicit + **dict)
            train_params = {k: v for k, v in params.items() if k != "seq_len"}

            model, history = train_model(
                train_data[0], train_data[1],
                val_data[0],   val_data[1],
                n_features=n_features,
                seq_len=seq_len,    # required by train_model — passed explicitly
                **train_params      # epochs, batch_size, learning_rate,
                                    # dropout, lstm_units, gru_units, conv_filters
            )

            # ── Validation metrics ─────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(val_data[0], dtype=torch.float32).to(DEVICE)
                preds = model(X_val_t).cpu().numpy()

            r2       = r2_score(val_data[1], preds)
            val_loss = history["val_loss"][-1]

            # Composite fitness: base (1/loss) scaled by R² reward
            fitness_val = (1.0 / (val_loss + 1e-9)) * (0.7 + 0.3 * max(0.0, r2))
            return fitness_val, r2

        except Exception as e:
            print(f"[DNA FAILURE]: {e}")
            return 0.0, -1.0

    # ── Genetic evolution loop ─────────────────────────────────────────────
    def evolve(self, pop_size=10, generations=10, mutation_rate=0.25, random_immigrants=0.15):
        """
        Evolves hyperparameters using:
          - Elitism           : best individual always carries forward
          - Random immigrants : fresh DNA injected each generation
          - Tournament crossover : uniform gene swap from top 40%
          - Mutation          : random gene re-roll at mutation_rate probability
        """
        population = [self._create_individual() for _ in range(pop_size)]

        for gen in range(generations):
            # ── Evaluate all individuals ───────────────────────────────────
            results = []
            for ind in population:
                fit, r2 = self._fitness(ind)
                results.append({"fitness": fit, "r2": r2, "params": ind})

            results.sort(key=lambda x: x["fitness"], reverse=True)
            best = results[0]
            print(
                f"Gen {gen + 1:>2} | "
                f"Best R²: {best['r2']:+.4f} | "
                f"Fitness: {best['fitness']:.6f} | "
                f"Params: {best['params']}"
            )

            # ── Build next generation ──────────────────────────────────────
            next_gen = [best["params"]]                              # 1. Elitism

            num_immigrants = max(1, int(pop_size * random_immigrants))
            for _ in range(num_immigrants):                          # 2. Immigrants
                next_gen.append(self._create_individual())

            top_k = max(2, int(pop_size * 0.4))                      # 3. Crossover + Mutation
            while len(next_gen) < pop_size:
                p1 = results[random.randint(0, top_k - 1)]["params"]
                p2 = results[random.randint(0, top_k - 1)]["params"]

                # Uniform crossover
                child = {
                    k: (p1[k] if random.random() > 0.5 else p2[k])
                    for k in self.param_grid
                }

                # Single-gene mutation
                if random.random() < mutation_rate:
                    mut_key = random.choice(list(self.param_grid.keys()))
                    child[mut_key] = random.choice(self.param_grid[mut_key])

                next_gen.append(child)

            population = next_gen

        return results[0]["params"]   # Champion hyperparameters


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import fetch_all_stocks_sequential

    TICKERS = [
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "ICICIBANK.NS",
        "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "ADANIENT.NS",
        "HINDUNILVR.NS", "BAJFINANCE.NS",
    ]

    print("Fetching stock data …")
    data_store = fetch_all_stocks_sequential(TICKERS, years=1)

    all_best_params = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"  Genetic Optimization  →  {ticker}")
        print(f"{'=' * 60}")

        stock_data = data_store[ticker]
        ga = GeneticOptimizer(stock_data["price_data"], stock_data["fundamentals"])
        best_params = ga.evolve(pop_size=10, generations=10)

        all_best_params[ticker] = best_params
        print(f"\n✔  Champion params for {ticker}:")
        for k, v in best_params.items():
            print(f"     {k:<16} = {v}")

    print(f"\n\n{'=' * 60}")
    print("  FULL OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    for ticker, params in all_best_params.items():
        print(f"\n{ticker}: {params}")
