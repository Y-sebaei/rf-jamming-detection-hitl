
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE    = "data/rfLoc1_6dbmJamming.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.30
# ────────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE).dropna()
X  = df.drop(columns=["label"])
y  = df["label"]

print(f"Dataset: {X.shape}  |  Classes: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ─── Models ─────────────────────────────────────────────────────────────────────
models = {
    "Random Forest (ours)": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features="sqrt",
        bootstrap=False,
        random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE
    ),
    # Linear models require feature scaling to converge reliably
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(max_iter=5000, dual=False, random_state=RANDOM_STATE)
        ))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
}

# ─── Train and evaluate ─────────────────────────────────────────────────────────
results = []

for name, model in models.items():
    print(f"Training: {name} ...", end="  ", flush=True)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
        "Train Time (s)": round(train_time, 3),
        "Infer Time (s)": round(infer_time, 4),
    })

    print(f"Acc={acc:.4f}  F1={f1:.4f}  [{train_time:.2f}s train]")

# ─── Results table ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

results_df.to_csv("data/model_comparison_results.csv", index=False)
print("\nResults saved to model_comparison_results.csv")

# ─── Detailed report for our model ──────────────────────────────────────────────
print("\nDetailed report for Random Forest (ours):")
rf_model = models["Random Forest (ours)"]
print(classification_report(y_test, rf_model.predict(X_test),
                             target_names=["Benign", "Jamming"]))

# ─── Bar chart ──────────────────────────────────────────────────────────────────
metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_names = results_df["Model"].tolist()
x = np.arange(len(model_names))
bar_width = 0.18

fig, ax = plt.subplots(figsize=(13, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = results_df[metric].tolist()
    bars = ax.bar(x + i * bar_width, vals, bar_width,
                  label=metric, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison: Jamming Detection", fontsize=13)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=150)
print("Chart saved to model_comparison.png")
plt.show()
