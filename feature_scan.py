import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load and prep
df = pd.read_excel("pgatour_players_stats.xlsx")

# Drop non-numerical IDs, URLs, all "Rank" columns, and any "Scoring Average" to prevent leakage
X = df.drop(
    [c for c in df.columns
     if ("Rank" in c)
     or ("Scoring Average" in c)
     or ("Lowest Round" in c)
     or (c in [
         "Player ID",
         "Player",
         "Profile URL",
         "Stats Page URL"
     ])],
    axis=1
)
y = df["Scoring Average (Adjusted) - Primary"]

# Clean target variable: replace '-' with NaN and convert to numeric
y = y.astype(str).replace('-', 'NaN')
y = pd.to_numeric(y, errors='coerce')

# Clean numeric columns
for col in X.columns:
    if X[col].dtype == 'object':
        # Remove dollar signs and commas, replace '-' with NaN
        X[col] = X[col].astype(str).str.replace('$', '').str.replace(',', '').replace('-', 'NaN')
        # Convert to numeric, keeping NaN values
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Remove rows where target variable is NaN
valid_indices = y.notna()
X = X[valid_indices]
y = y[valid_indices]

# Fill remaining NaN values with column means
X = X.fillna(X.mean())

# Sanitize feature names for XGBoost compatibility
sanitized_cols = []
for col in X.columns:
    new_col = (col
               .replace('[', '')
               .replace(']', '')
               .replace('<', 'lt')
               .replace('>', 'gt')
               .replace(',', '')
               .replace('(', '')
               .replace(')', ''))
    sanitized_cols.append(new_col)
X.columns = sanitized_cols

# Convert all data to float32 to ensure compatibility with XGBoost
X = X.astype('float32')
y = y.astype('float32')

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Baseline XGBoost
model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=10
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred = model.predict(X_test)
# Compute RMSE manually since this sklearn version does not support squared=False
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"\nTest RMSE: {rmse:.4f}")

# 4. Compute and display feature importances by weight and gain
booster = model.get_booster()
# Retrieve raw importance dicts
weight_dict = booster.get_score(importance_type='weight')
gain_dict   = booster.get_score(importance_type='gain')

# Convert to pandas Series and sort descending
weight_series = pd.Series(weight_dict).sort_values(ascending=False)
gain_series   = pd.Series(gain_dict).sort_values(ascending=False)

# Display the top 30 features by each metric
print("Top 30 features by weight (frequency of use):")
print(weight_series.head(30))

print("\nTop 30 features by gain (average loss reduction):")
print(gain_series.head(30))