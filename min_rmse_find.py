import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load and prep (same as feature_scan.py)
df = pd.read_excel("pgatour_players_stats.xlsx")

# Drop leakage columns
X = df.drop(
    [c for c in df.columns
     if ("Rank" in c)
     or ("Scoring Average" in c)
     or ("Lowest Round" in c)
     or ("Final Round Performance" in c)
     or ("Comcast Business TOUR TOP 10" in c)
     or (c in ["Player ID","Player","Profile URL","Stats Page URL"])],
    axis=1
)

# Target
y = df["Scoring Average (Adjusted) - Primary"].astype(str).replace('-', 'NaN')
y = pd.to_numeric(y, errors='coerce')

# Clean X columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = (X[col]
                  .astype(str)
                  .str.replace('$','')
                  .str.replace(',','')
                  .replace('-', 'NaN'))
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Filter valid rows
valid = y.notna()
X = X[valid].fillna(X.mean())
y = y[valid].astype('float32')

# Sanitize names
new_cols = []
for col in X.columns:
    new_cols.append(col.replace('[','').replace(']','')
                        .replace('<','lt').replace('>','gt')
                        .replace(',','').replace('(','').replace(')',''))
X.columns = new_cols
X = X.astype('float32')

# 2. Baseline train/test split & model to get gain importances
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
base_model = XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, early_stopping_rounds=10
)
base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Extract gain importances
booster = base_model.get_booster()
gain_dict = booster.get_score(importance_type='gain')
gain_series = pd.Series(gain_dict).sort_values(ascending=False)

# 3–5. Find optimal number of top-gain features by scanning N=1..min(50, total features)
max_features = min(50, len(gain_series))
results = []
for N in range(1, max_features + 1):
    features_N = gain_series.head(N).index.tolist()
    # Split data
    X_N = X[features_N]
    X_tr, X_te, y_tr, y_te = train_test_split(X_N, y, test_size=0.2, random_state=42)
    # Train model
    model_N = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=10
    )
    model_N.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    # Evaluate RMSE
    y_pred = model_N.predict(X_te)
    rmse = mean_squared_error(y_te, y_pred) ** 0.5
    results.append((N, rmse))
# Identify best N
best_N, best_rmse = min(results, key=lambda x: x[1])
print(f"Best feature count: {best_N} with Test RMSE: {best_rmse:.4f}")
# List the features for best N
print("Features for best N (feature: gain):")
for feat in gain_series.head(best_N).index:
    gain_value = gain_series[feat]
    print(f"- {feat}: {gain_value:.6f}")