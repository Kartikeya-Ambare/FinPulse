# # import numpy as np
# # import random
# # import copy
# # import torch
# # from ml_model import train_model, DEVICE
# # from feature_engineering import build_features, create_sequences

# # class GeneticOptimizer:
# #     def __init__(self, price_data, fundamentals, seq_len=30):
# #         raf = fundamentals.get("RAF", 1.0)
# #         df_feat = build_features(price_data, raf=raf)
        
# #         X, y, self.scaler_X, self.scaler_y, self.feat_cols, self.raw_ohlc = create_sequences(df_feat, seq_len)
        
# #         split = int(len(X) * 0.8)
# #         self.train_data = (X[:split], y[:split])
# #         self.val_data = (X[split:], y[split:])
# #         self.seq_len = seq_len
# #         self.n_features = len(self.feat_cols)

# #         self.param_grid = {
# #             'lstm_units':   [16, 32, 48, 64, 80, 96, 128, 160, 256],
# #             'gru_units':    [16, 32, 48, 64, 80, 96, 128, 160, 256],
# #             'conv_filters': [16, 32, 48, 64, 80, 96, 128, 192, 256],
    
# #             'dropout':      [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    
# #             'learning_rate':[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
    
# #             'batch_size':   [8, 16, 24, 32, 48, 64]
# #         }

# #     def _create_individual(self):
# #         """Randomly sample hyperparameters."""
# #         return {k: random.choice(v) for k, v in self.param_grid.items()}

# #     def _fitness(self, params):
# #         """Train model and return 1/Val_Loss (Higher fitness is better)."""
# #         try:
# #             # Shortened epochs for evolution speed
# #             model, history = train_model(
# #                 self.train_data[0], self.train_data[1],
# #                 self.val_data[0], self.val_data[1],
# #                 seq_len=self.seq_len,
# #                 n_features=self.n_features,
# #                 epochs=15, 
# #                 **params
# #             )
# #             val_loss = history['val_loss'][-1]
# #             return 1.0 / (val_loss + 1e-9)
# #         except Exception as e:
# #             print(f"Gene failed: {e}")
# #             return 0.0

# #     def evolve(self, pop_size=8, generations=5):
# #         # Initialize Population
# #         population = [self._create_individual() for _ in range(pop_size)]
        
# #         for gen in range(generations):
# #             print(f"\n--- Generation {gen + 1} ---")
            
# #             # Evaluate Fitness
# #             scores = []
# #             for individual in population:
# #                 fit_score = self._fitness(individual)
# #                 scores.append((fit_score, individual))
            
# #             # Sort by fitness (descending)
# #             scores.sort(key=lambda x: x[0], reverse=True)
# #             print(f"Best Fitness: {scores[0][0]:.4f} | Params: {scores[0][1]}")

# #             # Selection: Carry over top 2 (Elitism)
# #             next_gen = [scores[0][1], scores[1][1]]

# #             # Crossover & Mutation
# #             while len(next_gen) < pop_size:
# #                 parent1 = scores[random.randint(0, 3)][1]
# #                 parent2 = scores[random.randint(0, 3)][1]
                
# #                 # Crossover: Mix genes
# #                 child = {}
# #                 for k in self.param_grid.keys():
# #                     child[k] = parent1[k] if random.random() > 0.5 else parent2[k]
                
# #                 # Mutation: Random change
# #                 if random.random() < 0.2:
# #                     k = random.choice(list(self.param_grid.keys()))
# #                     child[k] = random.choice(self.param_grid[k])
                
# #                 next_gen.append(child)
            
# #             population = next_gen

# #         return scores[0][1] # Return champion hyperparameters
    





# import numpy as np
# import random
# import torch
# from sklearn.metrics import r2_score
# from ml_model import train_model, DEVICE
# from feature_engineering import build_features, create_sequences

# class GeneticOptimizer:
#     def __init__(self, price_data, fundamentals, seq_len=30):
#         raf = fundamentals.get("RAF", 1.0)
#         df_feat = build_features(price_data, raf=raf)
        
#         # Create sequences: returns X, y (log returns), scalers, and feature names
#         X, y, self.scaler_X, self.scaler_y, self.feat_cols, self.raw_ohlc = create_sequences(df_feat, seq_len)
        
#         split = int(len(X) * 0.8)
#         self.train_data = (X[:split], y[:split])
#         self.val_data = (X[split:], y[split:])
#         self.seq_len = seq_len
#         self.n_features = len(self.feat_cols)

#         # --- EXTENSIVE HYPERPARAMETER GRID ---
#         self.param_grid = {
#             "seq_len":        [20, 25, 30, 35, 40, 45, 50, 55, 60],
    
#     # epochs: Recommended range 80–150
#     "epochs":         [80, 90, 100, 110, 120, 130, 140, 150],
    
#     # batch_size: Guide options 16, 32, 64
#     "batch_size":     [16, 32, 64],
    
#     # learning_rate: Recommended range 0.0005–0.002
#     "learning_rate":  [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002],
    
#     # dropout: Higher for volatile stocks. Range: 0.10–0.40
#     "dropout":        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    
#     # lstm_units: Options 32, 64, 96, 128
#     "lstm_units":     [32, 64, 96, 128],
    
#     # gru_units: Options 32, 64, 96, 128
#     "gru_units":      [32, 64, 96, 128],
    
#     # conv_filters: Options 32, 64, 128
#     "conv_filters":   [32, 64, 128]
#         }

#     def _create_individual(self):
#         """Creates a chromosome using random choices from the expanded grid."""
#         return {k: random.choice(v) for k, v in self.param_grid.items()}

#     def _calculate_r2(self, model, X_val, y_val):
#         """Calculates the R2 score for the predicted log-returns."""
#         model.eval()
#         with torch.no_grad():
#             X_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
#             preds = model(X_t).cpu().numpy()
        
#         # We calculate R2 on the scaled log-returns directly
#         return r2_score(y_val, preds)

#     def _fitness(self, params):
#         """
#         Calculates fitness based on Loss and R2.
#         Higher R2 + Lower Loss = Superior DNA.
#         """
#         try:
#             model, history = train_model(
#                 self.train_data[0], self.train_data[1],
#                 self.val_data[0], self.val_data[1],
#                 seq_len=self.seq_len,
#                 n_features=self.n_features,
#                 epochs=15, # Quick evaluation
#                 **params
#             )
            
#             val_loss = history['val_loss'][-1]
#             r2 = self._calculate_r2(model, self.val_data[0], self.val_data[1])
            
#             # Penalize R2 if it is negative (model is worse than a mean baseline)
#             r2_score_adj = max(0, r2) 
            
#             # Composite Fitness: Weighted combination
#             # (1 / Loss) provides the base, + R2 reward
#             fitness_val = (1.0 / (val_loss + 1e-9)) * (0.7 + 0.3 * r2_score_adj)
#             return fitness_val, r2
#         except Exception as e:
#             return 0.0, -1.0

#     def evolve(self, pop_size=16, generations=10, mutation_rate=0.25, random_immigrants=0.15):
#         """
#         Evolves the population using Elitism, Crossover, Mutation, and 
#         Random Immigrants (Random Choice) for diversity.
#         """
#         # Initial Population: All Random Choices
#         population = [self._create_individual() for _ in range(pop_size)]
        
#         for gen in range(generations):
#             results = []
#             for ind in population:
#                 fit, r2 = self._fitness(ind)
#                 results.append({'fitness': fit, 'r2': r2, 'params': ind})
            
#             # Sort by fitness descending
#             results.sort(key=lambda x: x['fitness'], reverse=True)
            
#             best_ind = results[0]
#             print(f"Gen {gen+1} | Best R2: {best_ind['r2']:.4f} | Fitness: {best_ind['fitness']:.4f}")

#             # 1. Elitism: Carry best individual forward
#             next_gen = [best_ind['params']] 

#             # 2. Random Choice (Immigrants): Inject fresh DNA to prevent local optima
#             num_immigrants = int(pop_size * random_immigrants)
#             for _ in range(num_immigrants):
#                 next_gen.append(self._create_individual())

#             # 3. Crossover & Mutation
#             while len(next_gen) < pop_size:
#                 # Tournament: Pick 2 from top 40%
#                 parent1 = results[random.randint(0, int(pop_size * 0.4))]['params']
#                 parent2 = results[random.randint(0, int(pop_size * 0.4))]['params']
                
#                 # Crossover: Uniform swap
#                 child = {k: (parent1[k] if random.random() > 0.5 else parent2[k]) for k in self.param_grid}
                
#                 # Mutation: Random choice for one gene
#                 if random.random() < mutation_rate:
#                     mut_key = random.choice(list(self.param_grid.keys()))
#                     child[mut_key] = random.choice(self.param_grid[mut_key])
                
#                 next_gen.append(child)
            
#             population = next_gen

#         return results[0]['params']

# from data_loader import fetch_all_stocks_sequential

# tickers = ["TCS.NS","INFY.NS","HDFCBANK.NS","WIPRO.NS","ICICIBANK.NS","BHARTIARTL.NS","ITC.NS","SBIN.NS","ADANIENT.NS","HINDUNILVR.NS","BAJFINANCE.NS"]
# data_store = fetch_all_stocks_sequential(tickers, years=1)

# # 2. Optimize for a specific stock

# for i in tickers:
#     stock_data = data_store[i]
#     ga = GeneticOptimizer(stock_data["price_data"], stock_data["fundamentals"])

# # 3. Find best params
#     best_params = ga.evolve(pop_size=10, generations=10)
#     print(f"For Optimized Hyperparameters for {i}: {best_params}")








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
        
        # Grid strictly following your PARAMETER GUIDE
        self.param_grid = {
            "seq_len":        [20, 25, 30, 35, 40, 45, 50, 55, 60],
            "epochs":         [80, 90, 100, 110, 120, 130, 140, 150],
            "batch_size":     [16, 32, 64],
            "learning_rate":  [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002],
            "dropout":        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "lstm_units":     [32, 64, 96, 128],
            "gru_units":      [32, 64, 96, 128],
            "conv_filters":   [32, 64, 128]
        }

    def _create_individual(self):
        return {k: random.choice(v) for k, v in self.param_grid.items()}

    def _prepare_data_dynamic(self, seq_len):
        raf = self.fundamentals.get("RAF", 1.0)
        df_feat = build_features(self.price_data, raf=raf)
        X, y, _, _, _, _ = create_sequences(df_feat, seq_len=seq_len)
        split = int(len(X) * 0.8)
        return (X[:split], y[:split]), (X[split:], y[split:]), X.shape[2]

    def _fitness(self, params):
        try:
            # 1. Prepare data for this specific DNA's seq_len
            train_data, val_data, n_features = self._prepare_data_dynamic(params['seq_len'])
            
            # 2. Train the model
            # FIX: Removed 'seq_len=params["seq_len"]' because it is inside **params
            model, history = train_model(
                train_data[0], train_data[1],
                val_data[0], val_data[1],
                n_features=n_features,
                epochs=15, # Quick training for GA evolution
                **params
            )
            
            # 3. Validation and R2 score
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(val_data[0], dtype=torch.float32).to(DEVICE)
                preds = model(X_val_t).cpu().numpy()
            
            r2 = r2_score(val_data[1], preds)
            val_loss = history['val_loss'][-1]
            
            fitness_val = (1.0 / (val_loss + 1e-9)) * (0.7 + 0.3 * max(0, r2))
            return fitness_val, r2
            
        except Exception as e:
            print(f"[DNA FAILURE]: {e}")
            return 0.0, -1.0

    def evolve(self, pop_size=10, generations=10, mutation_rate=0.25, random_immigrants=0.15):
        population = [self._create_individual() for _ in range(pop_size)]
        
        for gen in range(generations):
            results = []
            for ind in population:
                fit, r2 = self._fitness(ind)
                results.append({'fitness': fit, 'r2': r2, 'params': ind})
            
            results.sort(key=lambda x: x['fitness'], reverse=True)
            best_ind = results[0]
            print(f"Gen {gen+1} | Best R2: {best_ind['r2']:.4f} | Fitness: {best_ind['fitness']:.4f}")

            next_gen = [best_ind['params']] 

            for _ in range(int(pop_size * random_immigrants)):
                next_gen.append(self._create_individual())

            while len(next_gen) < pop_size:
                p1 = results[random.randint(0, int(pop_size * 0.4))]['params']
                p2 = results[random.randint(0, int(pop_size * 0.4))]['params']
                
                child = {k: (p1[k] if random.random() > 0.5 else p2[k]) for k in self.param_grid}
                if random.random() < mutation_rate:
                    mut_key = random.choice(list(self.param_grid.keys()))
                    child[mut_key] = random.choice(self.param_grid[mut_key])
                
                next_gen.append(child)
            
            population = next_gen

        return results[0]['params']


from data_loader import fetch_all_stocks_sequential

# 1. Data Fetching
tickers = [
    "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "ICICIBANK.NS", 
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "ADANIENT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS"
]
data_store = fetch_all_stocks_sequential(tickers, years=1)

# 2. Sequential Optimization Loop
for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Starting Genetic Optimization for: {ticker}")
    print(f"{'='*60}")
    
    stock_data = data_store[ticker]
    
    # Initialize fixed GA
    ga = GeneticOptimizer(stock_data["price_data"], stock_data["fundamentals"])

    # Evolve the best hyperparameters
    best_params = ga.evolve(pop_size=10, generations=10)
    
    print(f"\nOPTIMIZATION COMPLETE for {ticker}:")
    print(best_params)