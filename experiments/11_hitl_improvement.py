import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─── Configuration ──────────────────────────────────────────────────────────────
LOC1_FILE  = "data/rfLoc1_6dbmJamming.csv"
LOC2_FILE  = "data/rfLoc2_6dbmJamming.csv"
LOC3_FILE  = "data/rfLoc3_6dbmJamming.csv"

RANDOM_STATE    = 42
TRAIN_FRAC      = 0.02    # fraction of Loc2+Loc3 used as initial training data
topfeats        = 9       # SHAP top features
CONF_LOWER      = 0.20    # uncertain band lower bound
CONF_UPPER      = 0.80    # uncertain band upper bound
MAX_HITL        = 800     # max expert corrections per round

# Expert accuracy levels to evaluate (1.0 = perfect, 0.5 = random chance)
EXPERTISE_LEVELS = [1.0, 0.80, 0.50]
# ────────────────────────────────────────────────────────────────────────────────

for f in [LOC1_FILE, LOC2_FILE, LOC3_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Dataset not found: {f}")

loc1 = pd.read_csv(LOC1_FILE).dropna()
loc2 = pd.read_csv(LOC2_FILE).dropna()
loc3 = pd.read_csv(LOC3_FILE).dropna()

print(f"Loaded  Loc1: {loc1.shape}  Loc2: {loc2.shape}  Loc3: {loc3.shape}")

# ── Training set: small sample of Loc2 + Loc3 (model has never seen Loc1) ──────
loc2_sample = loc2.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE)
loc3_sample = loc3.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE)
train_df    = pd.concat([loc2_sample, loc3_sample], ignore_index=True)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

# ── Loc1 split: HITL pool (50%) | final test (50%) ──────────────────────────────
loc1_X = loc1.drop(columns=["label"])
loc1_y = loc1["label"]

X_hitl_pool, X_final_test, y_hitl_pool, y_final_test = train_test_split(
    loc1_X, loc1_y,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=loc1_y
)

print(f"\nTraining set  : {len(X_train):>5} samples  "
      f"({int(TRAIN_FRAC*100)}% of Loc2 + {int(TRAIN_FRAC*100)}% of Loc3)")
print(f"HITL pool     : {len(X_hitl_pool):>5} samples  (Loc1 — unseen location)")
print(f"Final test    : {len(X_final_test):>5} samples  (Loc1 — held-out)")

# ── Baseline model (all 49 features) ────────────────────────────────────────────
rf_baseline = RandomForestClassifier(
    n_estimators=200, max_depth=5,
    max_features="sqrt", bootstrap=False,
    random_state=RANDOM_STATE
)
rf_baseline.fit(X_train, y_train)

before_acc = accuracy_score(y_final_test, rf_baseline.predict(X_final_test))
print(f"\n{'='*60}")
print(f"BEFORE HITL  (all 49 features, Loc1 final_test): {before_acc:.4f}")
print(f"{'='*60}")

# ── SHAP feature selection ────────────────────────────────────────────────────────
print("Computing SHAP importances on training data...")
explainer   = shap.TreeExplainer(rf_baseline)
shap_values = explainer.shap_values(X_train)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
shap_df = pd.DataFrame({
    "feature": X_train.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

top_features = shap_df.head(topfeats)["feature"].tolist()
print(f"\nTop {topfeats} SHAP-selected features:")
for rank, feat in enumerate(top_features, 1):
    score = shap_df.loc[shap_df["feature"] == feat, "mean_abs_shap"].values[0]
    print(f"  {rank}. {feat}  (mean |SHAP| = {score:.5f})")

# ── Reduced feature sets ──────────────────────────────────────────────────────────
X_train_red      = X_train[top_features]
X_hitl_pool_red  = X_hitl_pool[top_features]
X_final_test_red = X_final_test[top_features]

rf_red = RandomForestClassifier(
    n_estimators=200, max_depth=5,
    max_features="sqrt", bootstrap=False,
    random_state=RANDOM_STATE
)
rf_red.fit(X_train_red, y_train)

pre_hitl_acc = accuracy_score(y_final_test, rf_red.predict(X_final_test_red))
print(f"\nPre-HITL accuracy ({topfeats} features, Loc1 final_test): {pre_hitl_acc:.4f}")

# ── Uncertain sample indices (same pool for all expertise levels) ─────────────────
probs = rf_red.predict_proba(X_hitl_pool_red)
uncertain_indices = [
    i for i in range(len(X_hitl_pool_red))
    if CONF_LOWER < probs[i][1] < CONF_UPPER
]
print(f"\nUncertain samples in HITL pool : {len(uncertain_indices)} / {len(X_hitl_pool_red)}")

if len(uncertain_indices) > MAX_HITL:
    print(f"Capping at {MAX_HITL} corrections.")
    uncertain_indices = uncertain_indices[:MAX_HITL]

# ── Expert simulation function ────────────────────────────────────────────────────
def simulate_expert(true_label, expertise, rng):
    """Return the correct label with probability=expertise, else the flipped label."""
    if rng.random() < expertise:
        return int(true_label)
    return int(1 - true_label)

# ── Run HITL for each expertise level ────────────────────────────────────────────
results = {}

for expertise in EXPERTISE_LEVELS:
    label_pct = int(round(expertise * 100))
    print(f"\n{'='*60}")
    print(f"HITL — Expert accuracy: {label_pct}%")
    print(f"{'='*60}")

    rng = np.random.default_rng(RANDOM_STATE)

    corrected_X = []
    corrected_y = []

    for i in uncertain_indices:
        true_label      = y_hitl_pool.iloc[i]
        corrected_label = simulate_expert(true_label, expertise, rng)
        corrected_X.append(X_hitl_pool_red.iloc[i])
        corrected_y.append(corrected_label)

    print(f"Expert corrections applied: {len(corrected_X)}")

    corrected_X_df  = pd.DataFrame(corrected_X, columns=top_features)
    corrected_y_ser = pd.Series(corrected_y, dtype=int)

    X_train_updated = pd.concat(
        [X_train_red, corrected_X_df], ignore_index=True
    )
    y_train_updated = pd.concat(
        [y_train.reset_index(drop=True), corrected_y_ser], ignore_index=True
    )

    rf_updated = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        max_features="sqrt", bootstrap=False,
        random_state=RANDOM_STATE
    )
    rf_updated.fit(X_train_updated, y_train_updated)

    post_acc = accuracy_score(y_final_test, rf_updated.predict(X_final_test_red))
    results[label_pct] = post_acc

    print(f"Post-HITL accuracy ({label_pct}% expert): {post_acc:.4f}  "
          f"(improvement: {post_acc - pre_hitl_acc:+.4f})")
    print(classification_report(y_final_test, rf_updated.predict(X_final_test_red)))

# ── Summary table ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Before HITL   (49 features)       : {before_acc:.4f}")
print(f"Pre-HITL      ({topfeats} features)         : {pre_hitl_acc:.4f}")
for pct, acc in results.items():
    print(f"Post-HITL     ({pct}% expert accuracy) : {acc:.4f}  "
          f"({(acc - pre_hitl_acc)*100:+.1f} pp)")
print(f"{'='*60}")

# ── Comparison graph ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

labels    = [f"{pct}% Expert" for pct in results.keys()]
post_accs = list(results.values())
colors    = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red

bars = ax.bar(labels, post_accs, color=colors, alpha=0.85, width=0.45, zorder=3)

# Annotate each bar
for bar, acc in zip(bars, post_accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc*100:.2f}%",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

# Horizontal reference line
ax.axhline(before_acc, color="grey", linestyle=":",
           linewidth=1.2, label=f"Before HITL ({before_acc*100:.2f}%)")

ax.set_xlabel("Expert Accuracy Level", fontsize=12)
ax.set_ylabel("Post-HITL Accuracy (Loc1 test)", fontsize=12)
ax.set_title("Effect of Expert Accuracy Level on HITL Performance", fontsize=13)
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

plt.tight_layout()

# Save locally and also into thesis img folder if present
out_paths = ["figures/hitl_expertise_comparison.png"]
thesis_img = r"C:\Users\YEHIA KHALED\OneDrive\Desktop\Masters_Thesis_Template\img"
if os.path.isdir(thesis_img):
    out_paths.append(os.path.join(thesis_img, "figures/hitl_expertise_comparison.png"))

for p in out_paths:
    plt.savefig(p, dpi=150)
    print(f"Graph saved to {p}")

plt.show()
