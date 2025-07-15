# %% [markdown]
# PGACOACH XGBOOST PROJECT
# --------------------------------------------------

# %%
# If you haven't installed shap, uncomment:
# !pip install shap

# %%
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import KNNImputer
import itertools

# %% ------------------------------------------------
# 1 · Loading and prepping data
# --------------------------------------------------
df = pd.read_excel("pgatour_players_stats.xlsx")

# drop leakage columns
X = df.drop(
    [c for c in df.columns
     if ("Rank" in c) or ("Scoring Average" in c) or ("Lowest Round" in c)
     or ("Final Round Performance" in c) or ("Comcast Business TOUR TOP 10" in c)
     or ("Official Money" in c) or ("SG: Total" in c) or ("Par Breakers" in c)
     or (c in ["Player ID", "Player", "Profile URL", "Stats Page URL"])],
    axis=1
)

# target
y = df["Scoring Average (Adjusted) - Primary"].astype(str).replace('-', 'NaN')
y = pd.to_numeric(y, errors='coerce')

# clean object columns
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = (X[col].astype(str)
                           .str.replace("$", "")
                           .str.replace(",", "")
                           .replace("-", "NaN"))
        X[col] = pd.to_numeric(X[col], errors="coerce")

# retain valid rows
valid = y.notna()
X = X[valid]
y = y[valid].astype("float32")

# sanitise names
X.columns = (
    X.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace("<", "lt", regex=False)
      .str.replace(">", "gt", regex=False)
      .str.replace(",",  "", regex=False)
      .str.replace("(",  "", regex=False)
      .str.replace(")",  "", regex=False)
)

# interaction feature: GIR × putting
if "Greens in Regulation - Primary" in X.columns and "Putts per Rd" in X.columns:
    X["GIR_putts"] = X["Greens in Regulation - Primary"] * X["Putts per Rd"]
elif "Greens in Regulation - Primary" in X.columns and \
     "Putting Average - Primary" in X.columns:
    X["GIR_putts"] = X["Greens in Regulation - Primary"] * X["Putting Average - Primary"]

# drop fully-NaN cols, KNN-impute the rest
X = X.dropna(axis=1, how="all")
X = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(X),
    columns=X.columns
).astype("float32")

# %% ------------------------------------------------
# 2 · Baseline XGB (no leakage) & gain importance
# --------------------------------------------------
# 60 / 20 / 20 split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val,  y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

base_model = XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, early_stopping_rounds=10
)
base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

gain_series = (
    pd.Series(base_model.get_booster().get_score(importance_type="gain"))
      .sort_values(ascending=False)
)

# %% ------------------------------------------------
# 3 · Scan top-N gain features
# --------------------------------------------------
max_features = min(50, len(gain_series))
results = []
for N in range(1, max_features + 1):
    fset = gain_series.head(N).index.tolist()
    X_sub = X[fset]
    X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y, test_size=0.20, random_state=42)
    m = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=10
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    rmse = mean_squared_error(y_te, m.predict(X_te), squared=False)
    results.append((N, rmse))

best_N, best_rmse = min(results, key=lambda t: t[1])
best_features = gain_series.head(best_N).index.tolist()
print(f"Top-{best_N} gain features → RMSE {best_rmse:.4f}")

# %% ------------------------------------------------
# 4 · SHAP on best features
# --------------------------------------------------
X_train_best = X_train[best_features]
X_test_best  = X_test[best_features]

model_best = XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, early_stopping_rounds=10
)
model_best.fit(X_train_best, y_train,
               eval_set=[(X_val[best_features], y_val)],
               verbose=False)

explainer   = shap.Explainer(model_best, X_train_best)
shap_values = explainer(X_test_best)

shap.plots.beeswarm(shap_values, max_display=len(best_features))
shap.summary_plot(shap_values, X_test_best, plot_type="bar",
                  max_display=len(best_features))

baseline_rmse = mean_squared_error(
    y_test, np.full_like(y_test, y_train.mean()), squared=False)
print(f"Baseline (mean) RMSE on TEST: {baseline_rmse:.4f}")

# %% ------------------------------------------------
# 5 · PDP examples
# --------------------------------------------------
pdp_feats = [
    f for f in ["Bogey Avoidance - Primary", "Consecutive GIR - Primary"]
    if f in X_test_best.columns
]
if pdp_feats:
    PartialDependenceDisplay.from_estimator(
        model_best, X_test_best, pdp_feats,
        kind="average", grid_resolution=25
    )

# %% ------------------------------------------------
# 6 · Randomised hyper-parameter search via xgb.cv
# --------------------------------------------------
dtrain = xgb.DMatrix(X[best_features], label=y)

param_grid = {
    'max_depth':        [3, 4, 5],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 5, 10],
    'gamma':            [0, 0.1, 0.3],
    'reg_lambda':       [1, 5, 10],
    'reg_alpha':        [0, 0.1, 0.5]
}

n_iter = 30
rng = np.random.default_rng(42)
samples = rng.choice(list(itertools.product(*param_grid.values())),
                     size=n_iter, replace=False)

best_cv_rmse, best_params = np.inf, None
print(f"Random search over {n_iter} parameter sets…")
for tup in samples:
    params = dict(zip(param_grid.keys(), tup))
    params.update(objective="reg:squarederror", seed=42)

    cv_res = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5,
                    metrics="rmse", early_stopping_rounds=30,
                    seed=42, verbose_eval=False)

    mean_rmse = cv_res["test-rmse-mean"].iloc[-1]
    if mean_rmse < best_cv_rmse:
        best_cv_rmse, best_params = mean_rmse, params
        print(f"  New best {best_cv_rmse:.4f} with {best_params}")

print("\n==== FINAL BEST ====")
print(f"CV RMSE {best_cv_rmse:.4f}")
print(best_params)