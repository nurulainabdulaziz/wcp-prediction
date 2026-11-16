import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

# Set the folder where the split CSV files are saved.
print("\nFile Loading...")
split_folder = Path(r"data\split data")

# Check training files from newest to oldest.
train_files_default = sorted(
    split_folder.glob("latest_data_train_*.csv"),
    reverse=True,
)

for train_path in train_files_default:
    timestamp = train_path.stem.removeprefix("latest_data_train_")
    test_path = split_folder / f"latest_data_test_{timestamp}.csv"

    if test_path.is_file():
        break
else:
    raise FileNotFoundError(
        f"No matching training and test CSV files found in {split_folder}"
    )

print("\nPress Enter to use the default train data file, or paste a CSV path.")
print(f"Default training file: {train_path.name}")
file_input_train = input("Training file path: ").strip().strip('"')
train_path = Path(file_input_train) if file_input_train else train_path

print("\nPress Enter to use the default test data file, or paste a CSV path.")
print(f"Default test file: {test_path.name}")
file_input_test = input("Test file path: ").strip().strip('"')
test_path = Path(file_input_test) if file_input_test else test_path

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

total_samples = len(train) + len(test)
actual_train_pct = 100 * len(train) / total_samples

train_pct = int(round(actual_train_pct, -1))
test_pct = 100 - train_pct

print(f"Training file: {train_path.name}")
print(f"Training dataset: \033[32m{train.shape[0]} samples, {train.shape[1]} parameters\033[0m")
print(f"Test file: {test_path.name}")
print(f"Test dataset: \033[32m{test.shape[0]} samples, {test.shape[1]} parameters\033[0m")
print(f"Estimate split ratio (train:test): \033[32m{train_pct}:{test_pct}\033[0m")
print("\033[32mDone! \033[0m\n")


#Define input and output target
TARGET = "moisture_chl"

X_train = train.drop(columns=[TARGET, 'tlabel'])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET, 'tlabel'])
y_test = test[TARGET]

# Scale the selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model directly
model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print('Model Training: \033[32mDone! \033[0m\n')
# print(y_pred)

def percentage_error(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.where(y_test != 0, np.abs(y_pred - y_test) / np.abs(y_test) * 100, np.nan)

r2 = r2_score(y_test, y_pred) 
errors = percentage_error(y_test, y_pred)
absolute_error = np.abs(y_pred - y_test)
metrics = pd.DataFrame({
    'r2': [r2],
    'rmse': [np.sqrt(mean_squared_error(y_test, y_pred))],
    'mae': [mean_absolute_error(y_test, y_pred)],
    'mape_percent': [np.nanmean(errors)],
})

predictions = pd.DataFrame({
    "TX Label": test["tlabel"],
    "test CHL (%)": y_test,
    "pred CHL (%)": y_pred,
    "Absolute Error (percentage points)": absolute_error,
    "Error (%)": errors,
}, index=getattr(y_test, "index", None))

print("\nModel performance")
print(metrics.to_string(index=False))
print("\nActual vs predicted")
print(predictions.to_string(index=False))

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
    "Feature": X_train.columns,
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

#save model
saved_at = datetime.now().astimezone()
timestamp = saved_at.strftime("%Y%m%d_%H%M%S")

model_folder = Path("models").resolve()
model_folder.mkdir(parents=True, exist_ok=True)
model_path = model_folder / f"xgb_chl_model_{timestamp}.joblib"

model_bundle = {
    "model": model,
    "scaler": scaler,
    "input_features": X_train.columns.tolist(),
    "saved_at": saved_at.isoformat(timespec="seconds"),
}

print("\nModel Saving...")
joblib.dump(model_bundle, model_path)
print(f"\033[32mModel saved successfully to {model_path}\033[0m\n\n")





