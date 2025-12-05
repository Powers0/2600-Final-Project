import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# --------------------------
# 1. Load CSV Files
# --------------------------

circuits = pd.read_csv("circuits.csv")
lap_times = pd.read_csv("lap_times.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
sprint_results = pd.read_csv("sprint_results.csv")
pit_stops = pd.read_csv("pit_stops.csv")   # This one is critical

pit_stops = pit_stops[pit_stops["milliseconds"] < 30000]
results_clean = results.drop_duplicates(subset=["raceId", "driverId"])

# Merge pit stops with results to get constructor (team)
pit_stops = pit_stops.merge(results_clean[["raceId", "driverId", "constructorId"]], 
                        on=["raceId", "driverId"], how="left")

# Target is now the raw pit_stop time 
pit_stops = pit_stops.rename(columns={"milliseconds" : "pit_time"})

# --------------------------
# 3. Merge features from races + circuits + seasons
# --------------------------

# Merge race info
df = pit_stops.merge(races, on="raceId", how="left")

# Merge circuit info
df = df.merge(circuits, on="circuitId", how="left")


# Merge constructor info (team)
constructors = pd.read_csv("constructors.csv")
constructors = constructors.drop(columns=["url"], errors="ignore")
df = df.merge(constructors, on="constructorId", how="left")

# --------------------------
# 4. Add sprint results as features (optional)
# --------------------------
for col in ["points", "position"]:
    sprint_results[col] = sprint_results[col].replace({"\\N": None})
    sprint_results[col] = pd.to_numeric(sprint_results[col], errors="coerce")

sprint_features = sprint_results.groupby(["raceId", "constructorId"]).agg(
    sprint_points=("points", "mean"),
    sprint_pos=("position", "mean")
).reset_index()

df = df.merge(sprint_features, on=["raceId", "constructorId"], how="left")

# --------------------------
# 5. Feature Cleaning
# --------------------------

# Example numeric features
numeric_features = [
    "altitude",    # if present in circuits
    "lat", "lng",  # circuit location
    "round",
    "sprint_points",
    "sprint_pos"
]

"""# Example categorical features
categorical_features = [
    "name_x",      # race name
    "circuitRef",
    "constructorRef",
    "country"
]"""

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

