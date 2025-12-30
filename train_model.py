"""
Train models for the Ride Sharing Price Prediction assignment and save the best model pipeline.

Run:
    python train_model.py
This creates:
    ride_price_model.joblib
"""
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

DATA_PATH = "ride_sharing_dataset.csv"
OUT_PATH = "ride_price_model.joblib"
RANDOM_STATE = 42

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    return {
        "RMSE": rmse(y_test, preds),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    num_cols = ["distance_miles", "duration_minutes", "hour", "day_of_week", "temperature", "driver_rating"]
    cat_cols = ["weather", "pickup_location", "dropoff_location", "vehicle_type"]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        # moderate settings to keep training time reasonable
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            n_jobs=-1,
            max_depth=20,
            min_samples_leaf=2,
        ),
    }

    fitted = {}
    metrics = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics[name] = evaluate(pipe, X_test, y_test)

    # Voting Regressor
    voting = VotingRegressor([
        ("lr", models["Linear Regression"]),
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"]),
    ])
    voting_pipe = Pipeline([("preprocess", preprocess), ("model", voting)])
    voting_pipe.fit(X_train, y_train)
    fitted["Voting Regressor"] = voting_pipe
    metrics["Voting Regressor"] = evaluate(voting_pipe, X_test, y_test)

    # Choose best by RMSE
    best_name = sorted(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0][0]
    best_pipe = fitted[best_name]

    print("Model comparison (test set):")
    for name, m in metrics.items():
        print(f"- {name}: RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, R2={m['R2']:.3f}")
    print(f"\nBest model by RMSE: {best_name}")

    joblib.dump(best_pipe, OUT_PATH)
    print(f"Saved best model pipeline to: {OUT_PATH}")

if __name__ == "__main__":
    main()
