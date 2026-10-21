"""
Human-in-the-Loop (HITL) jamming detection.

Data split (no contamination):
  - X_train      (60%) : model training
  - X_hitl_pool  (20%) : uncertain samples shown to human expert
  - X_final_test (20%) : held-out evaluation set, NEVER touched during HITL

The corrected HITL samples are merged back into training before retraining.
The final accuracy is measured on X_final_test only.

INTERACTIVE / AUTO mode
  Set INTERACTIVE = True  to enter labels manually at the terminal.
  Set INTERACTIVE = False to simulate an expert who always provides the true label.
"""

import pandas as pd
import numpy as np
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_FILE   = "data/rfLoc1_6dbmJamming.csv"
RANDOM_STATE = 42
topfeats    = 5          # top SHAP features for the reduced model
CONF_LOWER  = 0.30       # uncertain if jamming probability is between
CONF_UPPER  = 0.70       # CONF_LOWER and CONF_UPPER
INTERACTIVE = False      # False = auto-simulate expert with true labels
# ───────────────────────────────────────────────────────────────────────────────

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

# ── 3-way split: train / hitl_pool / final_test ────────────────────────────────
# First cut: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)
# Second cut: split the 40% equally → 20% hitl_pool, 20% final_test
X_hitl_pool, X_final_test, y_hitl_pool, y_final_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"\nSplit sizes: train={len(X_train)}  "
      f"hitl_pool={len(X_hitl_pool)}  "
      f"final_test={len(X_final_test)}")

# ── Full-feature baseline on final_test ────────────────────────────────────────
rf_full = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)
rf_full.fit(X_train, y_train)

baseline_acc = accuracy_score(y_final_test, rf_full.predict(X_final_test))
print(f"\nBaseline accuracy (all features, final_test): {baseline_acc:.4f}")

# ── SHAP feature selection — trained on X_train only ──────────────────────────
explainer = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X_train)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

shap_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

top_features = shap_importance_df.head(topfeats)["feature"].tolist()

print(f"\nTop {topfeats} SHAP-selected features:")
for rank, f in enumerate(top_features, 1):
    score = shap_importance_df.loc[shap_importance_df["feature"] == f, "mean_abs_shap"].values[0]
    print(f"  {rank}. {f}  (mean |SHAP| = {score:.5f})")

# ── Reduced-feature model (trained on X_train only) ───────────────────────────
X_train_red      = X_train[top_features]
X_hitl_pool_red  = X_hitl_pool[top_features]
X_final_test_red = X_final_test[top_features]

rf_reduced = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)
rf_reduced.fit(X_train_red, y_train)

y_pred_red = rf_reduced.predict(X_final_test_red)
reduced_acc = accuracy_score(y_final_test, y_pred_red)
print(f"\nReduced-feature accuracy ({topfeats} features, final_test): {reduced_acc:.4f}")

# ── HITL loop — runs on hitl_pool only ────────────────────────────────────────
print("\n" + "=" * 60)
print("HUMAN-IN-THE-LOOP")
print("=" * 60)

probs = rf_reduced.predict_proba(X_hitl_pool_red)

corrected_X = []
corrected_y = []

uncertain_indices = [
    i for i in range(len(X_hitl_pool_red))
    if CONF_LOWER < probs[i][1] < CONF_UPPER
]

print(f"Uncertain samples (confidence {CONF_LOWER}–{CONF_UPPER}): "
      f"{len(uncertain_indices)} / {len(X_hitl_pool_red)}")

for i in uncertain_indices:
    confidence = probs[i][1]
    model_pred = rf_reduced.predict(X_hitl_pool_red.iloc[[i]])[0]
    true_label = y_hitl_pool.iloc[i]

    print(f"\n  Sample {i} | Pred: {model_pred} | Conf(jamming): {confidence:.3f}")
    print(f"  Features: {X_hitl_pool_red.iloc[i].to_dict()}")

    if INTERACTIVE:
        while True:
            raw = input("  Enter correct label (0=Benign, 1=Jamming): ").strip()
            if raw in ("0", "1"):
                corrected_label = int(raw)
                break
            print("  Invalid input — please enter 0 or 1.")
    else:
        # Simulate expert: always provide the true label
        corrected_label = int(true_label)
        print(f"  [AUTO] Expert label: {corrected_label}")

    corrected_X.append(X_hitl_pool_red.iloc[i])
    corrected_y.append(corrected_label)

# ── Retrain with corrected samples, evaluate on final_test ────────────────────
print("\n" + "-" * 60)

if len(corrected_X) > 0:
    print(f"Retraining with {len(corrected_X)} corrected samples "
          f"added to training set...")

    corrected_X_df = pd.DataFrame(corrected_X, columns=top_features)
    corrected_y_ser = pd.Series(corrected_y, dtype=int)

    X_train_updated = pd.concat(
        [X_train_red, corrected_X_df], ignore_index=True
    )
    y_train_updated = pd.concat(
        [y_train.reset_index(drop=True), corrected_y_ser], ignore_index=True
    )

    rf_updated = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features="sqrt",
        bootstrap=False,
        random_state=RANDOM_STATE
    )
    rf_updated.fit(X_train_updated, y_train_updated)

    # Evaluate on the held-out final_test — never seen during HITL
    y_pred_updated = rf_updated.predict(X_final_test_red)
    updated_acc = accuracy_score(y_final_test, y_pred_updated)

    print(f"\nPre-HITL  accuracy  (final_test): {reduced_acc:.4f}")
    print(f"Post-HITL accuracy  (final_test): {updated_acc:.4f}")
    print(f"Improvement: {updated_acc - reduced_acc:+.4f}")
    print("\nPost-HITL Classification Report:")
    print(classification_report(y_final_test, y_pred_updated))
else:
    print("No uncertain samples found. Model unchanged.")
    print(f"Final accuracy (final_test): {reduced_acc:.4f}")
