import pandas as pd
import numpy as np
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

topfeats = 9
RANDOM_STATE = 42
DATA_FILE = "data/rfLoc1_6dbmJamming.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

if "label" not in df.columns:
    raise ValueError("Missing 'label' column in dataset")

if df.isnull().any().any():
    df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset loaded:", X.shape)
print("Class distribution:\n", y.value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# --- Baseline model (all features) ---
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)
print(f"\nBaseline accuracy (all {X.shape[1]} features): {baseline_acc:.4f}")
print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred))

# --- SHAP feature importance (computed on training data only) ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# Handle both old SHAP (list) and new SHAP (3-D array) output formats
if isinstance(shap_values, list):
    # Old SHAP: list of arrays per class
    shap_vals = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    # New SHAP: shape (n_samples, n_features, n_classes)
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

if len(mean_abs_shap) != X_train.shape[1]:
    raise ValueError(
        f"SHAP dimension mismatch: got {len(mean_abs_shap)}, expected {X_train.shape[1]}"
    )

shap_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

top_features = shap_importance_df.head(topfeats)["feature"].tolist()

print(f"\nTop {topfeats} SHAP-selected features:")
for rank, f in enumerate(top_features, 1):
    score = shap_importance_df.loc[shap_importance_df["feature"] == f, "mean_abs_shap"].values[0]
    print(f"  {rank}. {f}  (mean |SHAP| = {score:.5f})")

# --- Reduced model (top-N features) ---
X_train_red = X_train[top_features]
X_test_red = X_test[top_features]

rf_reduced = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)
rf_reduced.fit(X_train_red, y_train)

y_pred_red = rf_reduced.predict(X_test_red)
reduced_acc = accuracy_score(y_test, y_pred_red)

print(f"\nReduced-feature accuracy ({topfeats} features): {reduced_acc:.4f}")
print(f"Accuracy change: {reduced_acc - baseline_acc:+.4f}")
print("\nReduced-Feature Classification Report:")
print(classification_report(y_test, y_pred_red))

# Save reduced dataset for downstream use
reduced_df = X[top_features].copy()
reduced_df["label"] = y.values
reduced_df.to_csv("data/rfLoc1_reduced.csv", index=False)
print("Reduced dataset saved to rfLoc1_reduced.csv")
