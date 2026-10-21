"""
SHAP vs Permutation Feature Importance Comparison
===================================================
Computes both SHAP and Permutation importance for the same trained
Random Forest model and compares them side-by-side.

Outputs:
  - Ranked feature table (both methods) printed to console
  - Spearman rank correlation between the two rankings
  - feature_importance_comparison.csv
  - feature_importance_comparison.png  (horizontal bar chart, top 15 features)
  - feature_importance_rank_scatter.png (rank agreement scatter plot)

Both methods use the same train/test split and the same trained model.
Permutation importance is evaluated on X_test (unbiased).
SHAP importance is computed on X_train (standard for TreeExplainer).
"""

import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE    = "data/rfLoc1_6dbmJamming.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.30
TOP_N        = 15    # features to show in bar chart
# ────────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE).dropna()
X  = df.drop(columns=["label"])
y  = df["label"]

print(f"Dataset: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ─── Train model ────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Model accuracy: {acc:.4f}\n")

# ─── SHAP importance (on training data) ─────────────────────────────────────────
print("Computing SHAP importance (on X_train)...")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

shap_importance = np.mean(np.abs(shap_vals), axis=0)

shap_df = pd.DataFrame({
    "feature":         X_train.columns,
    "shap_importance": shap_importance
}).sort_values("shap_importance", ascending=False).reset_index(drop=True)

shap_df["shap_rank"] = shap_df.index + 1

# ─── Permutation importance (on test data — unbiased) ───────────────────────────
print("Computing Permutation importance (on X_test, 20 repeats)...")
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature":          X.columns,
    "perm_importance":  perm_result.importances_mean,
    "perm_std":         perm_result.importances_std
}).sort_values("perm_importance", ascending=False).reset_index(drop=True)

perm_df["perm_rank"] = perm_df.index + 1

# ─── Merge both rankings ────────────────────────────────────────────────────────
comparison_df = shap_df.merge(perm_df, on="feature")
comparison_df["rank_diff"] = (
    comparison_df["shap_rank"] - comparison_df["perm_rank"]
)
comparison_df = comparison_df.sort_values("shap_rank")

# ─── Spearman rank correlation ───────────────────────────────────────────────────
rho, pval = spearmanr(comparison_df["shap_rank"], comparison_df["perm_rank"])

# ─── Same-split robustness check (removes train/test confound) ───────────────────
# The main comparison uses SHAP on train and permutation on test, which mixes a
# methodological difference with a train/test distribution difference. To isolate
# the methodological effect, recompute both rankings on the SAME split and report
# the Spearman correlation for each matched case.
print("\nComputing same-split robustness check...")

# SHAP on the TEST set
shap_values_test = explainer.shap_values(X_test)
if isinstance(shap_values_test, list):
    shap_vals_test = shap_values_test[1]
elif hasattr(shap_values_test, "ndim") and shap_values_test.ndim == 3:
    shap_vals_test = shap_values_test[:, :, 1]
else:
    shap_vals_test = shap_values_test
shap_imp_test = np.mean(np.abs(shap_vals_test), axis=0)
shap_rank_test = (
    pd.Series(shap_imp_test, index=X.columns)
    .rank(ascending=False)
)

# Permutation on the TRAIN set
perm_train = permutation_importance(
    rf, X_train, y_train, n_repeats=20,
    random_state=RANDOM_STATE, n_jobs=-1
)
perm_rank_train = (
    pd.Series(perm_train.importances_mean, index=X.columns)
    .rank(ascending=False)
)

# SHAP train ranks and perm test ranks aligned by feature
shap_rank_train = (
    pd.Series(shap_importance, index=X.columns).rank(ascending=False)
)
perm_rank_test = (
    pd.Series(perm_result.importances_mean, index=X.columns).rank(ascending=False)
)

rho_test_test, p_tt = spearmanr(shap_rank_test, perm_rank_test)
rho_train_train, p_trtr = spearmanr(shap_rank_train, perm_rank_train)

print(f"  rho (SHAP train vs Perm test, main) : {rho:.4f}  (p={pval:.4e})")
print(f"  rho (SHAP test  vs Perm test)       : {rho_test_test:.4f}  (p={p_tt:.4e})")
print(f"  rho (SHAP train vs Perm train)      : {rho_train_train:.4f}  (p={p_trtr:.4e})")
print("  --> If the matched-split rho values are close to the main rho, the weak")
print("      correlation is a genuine methodological difference, not a split artefact.")

# ─── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("SHAP vs PERMUTATION IMPORTANCE — FULL FEATURE RANKING")
print("=" * 85)
print(f"{'Feature':<30} {'SHAP Rank':>9} {'SHAP Score':>11} "
      f"{'Perm Rank':>9} {'Perm Score':>11} {'Rank Diff':>10}")
print("-" * 85)
for _, row in comparison_df.iterrows():
    diff_str = f"{int(row['rank_diff']):+d}"
    print(f"{row['feature']:<30} {int(row['shap_rank']):>9} "
          f"{row['shap_importance']:>11.5f} "
          f"{int(row['perm_rank']):>9} {row['perm_importance']:>11.5f} "
          f"{diff_str:>10}")

print("=" * 85)
print(f"\nSpearman rank correlation (rho): {rho:.4f}  (p={pval:.4e})")
if rho > 0.8:
    print("Interpretation: STRONG agreement between SHAP and Permutation rankings.")
elif rho > 0.6:
    print("Interpretation: MODERATE agreement between SHAP and Permutation rankings.")
else:
    print("Interpretation: WEAK agreement — methods highlight different features.")

# ─── Features where rankings disagree the most ──────────────────────────────────
print("\nTop 5 largest ranking disagreements:")
top_disagree = comparison_df.reindex(
    comparison_df["rank_diff"].abs().nlargest(5).index
)
for _, row in top_disagree.iterrows():
    print(f"  {row['feature']:<30}  "
          f"SHAP rank {int(row['shap_rank']):>2}  vs  "
          f"Perm rank {int(row['perm_rank']):>2}  "
          f"(diff = {int(row['rank_diff']):+d})")

# ─── Save CSV ───────────────────────────────────────────────────────────────────
comparison_df.to_csv("results/feature_importance_comparison.csv", index=False)
print("\nFull comparison saved to feature_importance_comparison.csv")

# ─── Plot 1: Side-by-side horizontal bar chart (top N features each) ────────────
top_shap = shap_df.head(TOP_N)
top_perm = perm_df.head(TOP_N)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# SHAP bar chart
axes[0].barh(
    top_shap["feature"][::-1],
    top_shap["shap_importance"][::-1],
    color="#4C72B0", alpha=0.85
)
axes[0].set_xlabel("Mean |SHAP value|", fontsize=11)
axes[0].set_title(f"Top {TOP_N} Features — SHAP Importance\n(computed on training data)",
                  fontsize=11)
axes[0].grid(axis="x", linestyle="--", alpha=0.5)

# Permutation bar chart
axes[1].barh(
    top_perm["feature"][::-1],
    top_perm["perm_importance"][::-1],
    xerr=top_perm["perm_std"][::-1],
    color="#DD8452", alpha=0.85, capsize=3
)
axes[1].set_xlabel("Mean accuracy decrease", fontsize=11)
axes[1].set_title(f"Top {TOP_N} Features — Permutation Importance\n(computed on test data, 20 repeats)",
                  fontsize=11)
axes[1].grid(axis="x", linestyle="--", alpha=0.5)

fig.suptitle(
    f"SHAP vs Permutation Feature Importance\n"
    f"Spearman rho = {rho:.3f}  (p = {pval:.2e})",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig("figures/feature_importance_comparison.png", dpi=150, bbox_inches="tight")
print("Bar chart saved to feature_importance_comparison.png")

# ─── Plot 2: Rank agreement scatter ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 7))

ax2.scatter(
    comparison_df["shap_rank"],
    comparison_df["perm_rank"],
    alpha=0.7, s=60, color="#55A868"
)

# Annotate top 10 by SHAP rank
for _, row in comparison_df[comparison_df["shap_rank"] <= 10].iterrows():
    ax2.annotate(
        row["feature"],
        xy=(row["shap_rank"], row["perm_rank"]),
        xytext=(4, 2), textcoords="offset points",
        fontsize=7, alpha=0.9
    )

# Perfect agreement diagonal
max_rank = len(comparison_df)
ax2.plot([1, max_rank], [1, max_rank], "r--", alpha=0.5, label="Perfect agreement")

ax2.set_xlabel("SHAP Rank", fontsize=11)
ax2.set_ylabel("Permutation Rank", fontsize=11)
ax2.set_title(
    f"Feature Rank Agreement: SHAP vs Permutation\n"
    f"Spearman rho = {rho:.3f}  (p = {pval:.2e})",
    fontsize=12
)
ax2.legend(fontsize=9)
ax2.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/feature_importance_rank_scatter.png", dpi=150)
print("Rank scatter plot saved to feature_importance_rank_scatter.png")
plt.show()
