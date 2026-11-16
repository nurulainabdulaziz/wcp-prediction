import pandas as pd 
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.20 # train 80%: test 20%
RANDOM_STATE = 42

# Load the dataset
file_default = r"data\latest_data.xlsx"

print("\nPress Enter to use the default file, or paste a file path.")
print("\nDefault file: ", Path(file_default).name)
file_input = input("File path: ").strip().strip('"')
file_path = Path(file_input or file_default)

if file_path.suffix.lower() == ".csv":
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

print(f"Loaded file: {file_path.name}")
print(f"Dataset shape: {df.shape[0]} samples, {df.shape[1]} parameters")

print("\nFile Loading...")
try:
    df = pd.read_excel(file_path)
except ValueError:
    # Fall back to CSV when the selected file has a .csv extension.
    if Path(file_path).suffix.lower() != ".csv":
        raise
    df = pd.read_csv(file_path)

print("\033[32mDone!\033[0m")

def ask_to_split():
    while True:
        answer = input("\nDo you want to split the data? (yes/no): ").strip().lower()

        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False

        print("Please enter yes or no.")


if not ask_to_split():
    print("\nData splitting skipped. Filling missing numeric values...")

    df = df.fillna(df.mean(numeric_only=True))

    output_folder = file_path.parent / "imputed data"
    output_folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"{file_path.stem}_imputed_{timestamp}.csv"

    df.to_csv(output_path, index=False)

    print("\033[32mDone! Imputed dataset saved.\033[0m")
    print(f"Saved to: {output_path.resolve()}")
    raise SystemExit(0)

# Drop unwanted parameter columns
print(f"\nDrop unwanted parameters ...")
columns_to_drop = ['moisture_ch', 'moisture_cl', 'BV_2FOL', 'BV_5HMF', 
                   'BV_2ACF', 'BV_5MEF']

df1 = df.drop(
    columns=columns_to_drop,
    errors='ignore'
)
print(f"Latest dataset shape: {df1.shape[0]} samples, \033[31m{df1.shape[1]} parameters\033[0m")
print("\033[32mDone!\033[0m")

# Split data into train:test
print(f"\nSplit data into train and test ...")
print(f"\nPress Enter if use default setting train:test(80:20)")
test_input = input("Input test set percentage %: ").strip()

size = float(test_input) / 100 if test_input else TEST_SIZE

train, test = train_test_split(
    df1,
    test_size=size, # set as default 80:20
    random_state=RANDOM_STATE, # RANDOM_STATE makes the split reproducible.
    shuffle=True
)

print("\033[32mDone! Dataset split into train and test sets.\033[0m\n")
print(f"Requested split ratio (train:test): {100 * (1 - size):.0f}:{100 * size:.0f}")
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

# Count parameters with at least one missing value
n_missing_parameters = df.isna().any(axis=0).sum()
print(f"\nParameters with missing values: {n_missing_parameters}")

#Filled missing value
train_means = train.mean(numeric_only=True)
train_filled = train.fillna(train_means)
test_filled = test.fillna(train_means)

print("\033[32mDone! Missing values have been filled in.\033[0m\n")

# Save the datasets next to the input file.
output_folder = file_path.parent / "split data"
output_folder.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

train_path = output_folder / f"{file_path.stem}_train_{timestamp}.csv"
test_path = output_folder / f"{file_path.stem}_test_{timestamp}.csv"

train_filled.to_csv(train_path, index=False)
test_filled.to_csv(test_path, index=False)

print(f"Training file: {train_path.resolve()}")
print(f"Test file: {test_path.resolve()}")
print("\033[32mDone! Training and test files saved.\033[0m\n\n")




