import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


TARGET = "moisture_chl"

# Load the latest timestamped model.
model_folder = Path("models")
model_files = sorted(model_folder.glob("xgb_chl_model_*.joblib"), reverse=True)

if not model_files:
    raise FileNotFoundError(f"No saved model found in {model_folder.resolve()}")

model_path = model_files[0]
model_bundle = joblib.load(model_path)

# Load the latest timestamped test CSV.
split_folder = Path("data") / "split data"
test_files = sorted(split_folder.glob("latest_data_test_*.csv"), reverse=True)

if not test_files:
    raise FileNotFoundError(f"No test CSV found in {split_folder.resolve()}")

test_path = test_files[0]
test = pd.read_csv(test_path)

print(f"Model file: {model_path.name}")
print(f"Model saved at: {model_bundle['saved_at']}")
print(f"Test file: {test_path.name}")
print(f"Test samples: {len(test)}")

# Match the training feature order and apply the saved scaler.
X_test = test.loc[:, model_bundle["input_features"]]
y_test = test[TARGET]
X_test_scaled = model_bundle["scaler"].transform(X_test)
model = model_bundle["model"]
y_pred = model.predict(X_test_scaled)

def percentage_error(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.full(y_test.shape, np.nan, dtype=float)
    np.divide(
        100 * np.abs(y_pred - y_test),
        np.abs(y_test),
        out=errors,
        where=y_test != 0,
    )
    return errors

# Evaluate the predictions against measured CHL.
r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan
errors = percentage_error(y_test, y_pred)
absolute_error = np.abs(y_pred - y_test)
valid_errors = errors[~np.isnan(errors)]

metrics = pd.DataFrame({
    "r2": [r2],
    "rmse": [np.sqrt(mean_squared_error(y_test, y_pred))],
    "mae": [mean_absolute_error(y_test, y_pred)],
    "mape_percent": [valid_errors.mean() if valid_errors.size else np.nan],
})

predictions = pd.DataFrame({
    "TX Label": test["tlabel"],
    "test CHL (%)": y_test,
    "pred CHL (%)": y_pred,
    "Absolute Error (percentage points)": absolute_error,
    "Error (%)": errors,
})

print("\nModel performance")
print(metrics.round(4).to_string(index=False))
print("\nActual vs predicted")
print(predictions.round(4).to_string(index=False))

FEATURES = model_bundle["input_features"]

# Measure the signed change in test MAE after shuffling each feature.
result = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)

mean_mae_change = result.importances_mean
positive_increases = mean_mae_change.clip(min=0)
total_increase = positive_increases.sum()

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean MAE change (percentage points)": mean_mae_change,
    "Importance (%)": (
        100 * positive_increases / total_increase if total_increase > 0 else 0.0
    ),
    "Action": np.where(mean_mae_change < 0, "Suggested to remove", ""),
}).sort_values("Mean MAE change (percentage points)", ascending=False)

print("\nPermutation feature importance")
print(importance.to_string(
    index=False,
    formatters={
        "Mean MAE change (percentage points)": "{:+.4f}".format,
        "Importance (%)": "{:.2f}%".format,
    },
))
print("\n\033[32mDone processing ...\033[0m\n")