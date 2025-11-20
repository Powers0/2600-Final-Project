# random forest file 
"""
Starter pipeline for predicting F1 pit stop duration with RandomForestRegressor.

Dependencies:
  pip install pandas numpy scikit-learn joblib

Assumptions:
  - You have CSVs like: circuits.csv, races.csv, lap_times.csv, seasons.csv, pit_stops.csv (optional)
  - pit_stops.csv contains a duration (seconds / milliseconds) column named 'milliseconds' or 'duration'
  - If no pit_stops.csv, code attempts to extract pit info from lap_times.csv (common dataset differences)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.simplefilter("ignore")


def load_csv_if_exists(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def load_data(folder="data"):
    # load common files (change folder as needed)
    files = {}
    for fn in ["circuits.csv", "races.csv", "lap_times.csv", "seasons.csv", "pit_stops.csv"]:
        files[fn] = load_csv_if_exists(Path(folder) / fn)
    return files


def prepare_target_and_join(files):
    """
    Returns merged DataFrame with target column 'pit_duration_ms' (milliseconds).
    Strategy:
      1. If pit_stops.csv exists and contains milliseconds/duration, use it as target.
      2. Otherwise try to extract pit info from lap_times.csv (if it has pit_stop column or similar).
    """
    pit = files.get("pit_stops.csv")
    laps = files.get("lap_times.csv")
    races = files.get("races.csv")
    circuits = files.get("circuits.csv")
    seasons = files.get("seasons.csv")

    if pit is not None:
        df = pit.copy()
        # normalize target name
        if "milliseconds" in df.columns:
            df["pit_duration_ms"] = df["milliseconds"]
        elif "duration" in df.columns:
            df["pit_duration_ms"] = df["duration"]
        else:
            raise ValueError("pit_stops.csv found but no 'milliseconds' or 'duration' column present.")
        # keep relevant columns
        # Typical columns: raceId, driverId, stop (pit stop number), pit_duration_ms, team/constructorId may be missing
    else:
        # fallback: try to derive pit durations from lap_times if it contains pit info
        if laps is None:
            raise FileNotFoundError("Neither pit_stops.csv nor lap_times.csv found in the provided folder.")
        df = laps.copy()
        # Example heuristics (dataset-specific):
        # Some lap_times datasets include a 'pit' indicator or 'pit_stop' column; others include 'pit_time' etc.
        if "pit_stop" in df.columns and "pit_stop_duration" in df.columns:
            df = df[df["pit_stop"] == 1].copy()
            df["pit_duration_ms"] = df["pit_stop_duration"]
        else:
            # as a last resort, try to use 'milliseconds' on lap where lap number went up and lap time very large
            # NOTE: this is sketchy; prefer actual pit_stops.csv
            possible = df[df["milliseconds"] > 10000]  # arbitrary large lap in ms
            possible = possible.copy()
            possible["pit_duration_ms"] = possible["milliseconds"]
            df = possible

    # merge race info (e.g., circuitId, year) to use season/race-level features
    if races is not None and "raceId" in df.columns:
        races_small = races[["raceId", "year", "circuitId", "name"]].rename(columns={"name":"race_name"})
        df = df.merge(races_small, on="raceId", how="left")

    # merge circuits (e.g., country, location) if available
    if circuits is not None and "circuitId" in df.columns:
        circuits_small = circuits[["circuitId", "country", "location", "name"]].rename(columns={"name":"circuit_name"})
        df = df.merge(circuits_small, on="circuitId", how="left")

    # optionally merge seasons
    if seasons is not None and "year" in df.columns:
        # seasons.csv may have year -> season metadata
        seasons_small = seasons.copy()
        if "year" in seasons_small.columns:
            df = df.merge(seasons_small, on="year", how="left")

    # common cleanup: cast target numeric, drop rows with missing target
    df["pit_duration_ms"] = pd.to_numeric(df["pit_duration_ms"], errors="coerce")
    df = df.dropna(subset=["pit_duration_ms"])
    return df


def feature_engineering(df):
    """
    Create/clean features. This is dataset-specific; adapt these as needed.
    Example features:
      - race year
      - circuit country / name
      - driverId, constructorId (one-hot or target encode)
      - lap number, stop number
      - tyre type (if available)
      - session/weather indicators (if available)
    """
    df = df.copy()
    # Example safe features -- check existance before referencing
    df["year"] = df.get("year", df.get("race_year", np.nan))
    # If pit stop number exists, use it
    if "stop" in df.columns:
        df["stop_num"] = pd.to_numeric(df["stop"], errors="coerce")
    # Convert strings to categories to reduce memory
    for col in ["race_name", "circuit_name", "country", "location", "driverId", "constructorId", "team", "status"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Add basic numeric features if present
    for col in ["lap", "position", "milliseconds", "time"]:
        if col in df.columns and df[col].dtype.kind in "iufc":
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce")
    # Example derived features:
    if "year" in df.columns:
        df["is_recent"] = (df["year"] >= df["year"].max()).astype(int)
    return df


def build_and_train(df, target_col="pit_duration_ms", group_col="year", test_size=0.2, random_state=42):
    # choose features
    exclude = {target_col, "milliseconds", "time"}  # drop raw duplicates or strings that won't be used
    candidate_features = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]  # numeric
    # also include some categorical columns
    categorical_feats = [c for c in df.columns if str(df[c].dtype).startswith("category")]

    # but we'll be explicit: prefer these common columns if present
    numeric_features = [c for c in ["lap_num", "stop_num", "position_num", "is_recent", "lap", "milliseconds"] if c in df.columns and c not in exclude]
    # fallback if numeric_features empty:
    if not numeric_features:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [c for c in numeric_features if c != target_col]

    # categorical_features:
    cat_features = [c for c in ["constructorId", "driverId", "country", "circuit_name", "race_name", "team"] if c in df.columns]

    X = df[numeric_features + cat_features].copy()
    y = df[target_col].values

    # Group-aware split: ensure whole seasons/races in train/test by using GroupShuffleSplit if group_col present
    groups = df[group_col] if group_col in df.columns else None
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, cat_features)
    ], remainder="drop")

    # Random forest pipeline
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    pipe = Pipeline([
        ("pre", preprocessor),
        ("rf", rf)
    ])

    # Optional: small grid search; expand when you have more time/data
    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [10, 20, None],
        "rf__min_samples_leaf": [1, 5]
    }
    gcv = GridSearchCV(pipe, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1)
    print("Starting GridSearchCV (this may take a while depending on data size)...")
    gcv.fit(X_train, y_train)
    print("Best params:", gcv.best_params_)

    # evaluate on test
    best = gcv.best_estimator_
    y_pred = best.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }
    print("Test metrics:", metrics)

    # feature importance: need to get feature names after preprocessing
    feature_names = []
    # numeric names first
    feature_names.extend(numeric_features)
    # expand OHE categorical names
    ohe = best.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
    if cat_features:
        cat_ohe_names = ohe.get_feature_names_out(cat_features)
        feature_names.extend(cat_ohe_names.tolist())

    importances = best.named_steps["rf"].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
    print("Top feature importances:")
    print(fi)

    return best, metrics, fi, (X_test, y_test, y_pred)


def save_model(model, path="models/rf_pit_predictor.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Example usage
    files = load_data(folder="data")  # adjust path
    df = prepare_target_and_join(files)
    df = feature_engineering(df)
    model, metrics, feature_importances, test_info = build_and_train(df, target_col="pit_duration_ms", group_col="year")
    save_model(model, path="models/rf_pit_predictor.joblib")
