import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# --------------------------
# 1. Load CSV Files
# --------------------------

pit_stops = pd.read_csv("pit_stops_with_alt.csvv")   # This one is critical

# Example numeric features
numeric_features = [
    "alt",    # if present in circuits
    "stop",
    "lap"
]


# Keep only columns that exist
numeric_features = [f for f in numeric_features if f in df.columns]
#categorical_features = [f for f in categorical_features if f in df.columns]

# Drop rows missing target
df = df.dropna(subset=["pit_time"])

# --------------------------
# 6. One-Hot Encode categoricals
# --------------------------

"""encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[categorical_features])"""

# Build feature matrix
X = df[numeric_features].fillna(0).values
#X = np.hstack([X_num, X_cat])

y = df["pit_time"].values

# --------------------------
# 7. Train Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 8. Fit the Random Forest
# --------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------
# 9. Evaluate
# --------------------------

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Feature importances (optional)
# Reverse-map the encoded feature names
"""encoded_feature_names = (
    numeric_features + 
    list(encoder.get_feature_names_out(categorical_features))
)"""

importances = pd.Series(model.feature_importances_, index=numeric_features)
print(importances.sort_values(ascending=False).head(20))

