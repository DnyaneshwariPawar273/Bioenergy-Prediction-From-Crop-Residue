import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- IMPORT YOUR PREPROCESSING FUNCTIONS ----------------
# Must match updated OneHotEncoder(sparse_output=False)
from data_preprocessing import (
    load_dataset,
    describe_numeric,
    train_test_split_processed
)

# ---------------- DEFAULT PATHS ----------------
DEFAULT_CSV = r"C:\Users\Dnyaneshwari Pawar\OneDrive\Desktop\Bioenergy_prediction_project\Cleaned_Bioenergy_Dataset_.csv"
MODEL_OUT = os.path.join(os.path.dirname(__file__), "models", "model.pkl")


# ---------------- METRIC FUNCTION ----------------
def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------------- TRAIN FUNCTION ----------------
def train(csv_path: str = DEFAULT_CSV, out_path: str = MODEL_OUT):

    # Load dataset
    df = load_dataset(csv_path)
    print("Dataset loaded:", df.shape)
    print(describe_numeric(df))   # Prints summary statistics

    # Train-test split + preprocessing
    X_train, X_test, y_train, y_test, preprocessor = train_test_split_processed(df)

    # ========== BASELINE MODEL ==========
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr, rmse_lr, r2_lr = eval_metrics(y_test, y_pred_lr)

    # ========== RANDOM FOREST ==========
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf, rmse_rf, r2_rf = eval_metrics(y_test, y_pred_rf)

    # Choose best model based on RMSE
    best_model = rf if rmse_rf <= rmse_lr else lr
    best_name = "RandomForest" if best_model is rf else "LinearRegression"

    best_scores = (
        (mae_rf, rmse_rf, r2_rf) if best_model is rf else (mae_lr, rmse_lr, r2_lr)
    )

    # Pack everything for saving
    bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "metrics": {
            "LinearRegression": {"mae": mae_lr, "rmse": rmse_lr, "r2": r2_lr},
            "RandomForest": {"mae": mae_rf, "rmse": rmse_rf, "r2": r2_rf},
            "best": {
                "name": best_name,
                "mae": best_scores[0],
                "rmse": best_scores[1],
                "r2": best_scores[2],
            },
        },
        "features": {
            "categorical": ["Crop_Type"],
            "numeric": [
                "Yield_t_ha",
                "Residue_to_Product_Ratio",
                "Residue_Quantity_t",
                "Moisture_%",
                "Calorific_Value_MJ_kg",
            ],
        },
    }

    # Save best model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(bundle, out_path)

    print("\n============================")
    print(f" MODEL SAVED → {out_path}")
    print("============================")
    print(" BEST MODEL:", best_name)
    print(" MAE:", best_scores[0])
    print(" RMSE:", best_scores[1])
    print(" R² :", best_scores[2])
    print("============================\n")


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bioenergy Prediction Model")

    parser.add_argument("--data", type=str, default=DEFAULT_CSV, help="CSV dataset path")
    parser.add_argument("--out", type=str, default=MODEL_OUT, help="Output model path")

    args = parser.parse_args()
    train(args.data, args.out)
