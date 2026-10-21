import pandas as pd
import os
import sys

# ─── User settings ─────────────────────────────────────────────────────────────
INPUT_CSV   = "MergedLoc3.csv"
OUTPUT_CSV  = "featuresLoc3.csv"
LABEL_VALUE = 0     # 0 = normal, 1 = jamming
WINDOW_SIZE = 128
STEP_SIZE   = 128
# ───────────────────────────────────────────────────────────────────────────────

SPECTRAL_FEATURES = [
    "noise",
    "max_magnitude",
    "total_gain_db",
    "base_pwr_db",
    "rssi",
    "relpwr_db",
    "avgpwr_db"
]

if not os.path.exists(INPUT_CSV):
    sys.exit(f"Error: input file not found — {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {df.shape}")

# Drop frequency column (not used for classification per paper methodology)
if "freq1" in df.columns:
    df.drop(columns=["freq1"], inplace=True)

# Validate required columns
missing = [c for c in SPECTRAL_FEATURES if c not in df.columns]
if missing:
    sys.exit(f"Error: missing columns in input CSV: {missing}")

# Drop rows with NaN values before windowing
n_before = len(df)
df = df[SPECTRAL_FEATURES].dropna()
if len(df) < n_before:
    print(f"Warning: dropped {n_before - len(df)} rows containing NaN values")

if len(df) < WINDOW_SIZE:
    sys.exit(f"Error: dataset too small ({len(df)} rows) for window size {WINDOW_SIZE}")


def extract_stats(window: pd.DataFrame) -> dict:
    feats = {}
    for col in SPECTRAL_FEATURES:
        feats[f"{col}_min"]  = window[col].min()
        feats[f"{col}_max"]  = window[col].max()
        feats[f"{col}_mean"] = window[col].mean()
        feats[f"{col}_std"]  = window[col].std()
        feats[f"{col}_q75"]  = window[col].quantile(0.75)
        feats[f"{col}_q50"]  = window[col].quantile(0.50)
        feats[f"{col}_q25"]  = window[col].quantile(0.25)
    return feats


rows = []
for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
    window = df.iloc[start:start + WINDOW_SIZE]
    feats = extract_stats(window)
    feats["label"] = LABEL_VALUE
    rows.append(feats)

if not rows:
    sys.exit("Error: no windows extracted — check WINDOW_SIZE and STEP_SIZE")

features_df = pd.DataFrame(rows)

# Validate no NaNs propagated (e.g. from std of constant window)
if features_df.isnull().any().any():
    print(f"Warning: NaN values in output — filling std NaN with 0")
    features_df = features_df.fillna(0)

features_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")
print(f"Feature shape: {features_df.shape}  "
      f"({len(rows)} windows × {features_df.shape[1]} columns)")
