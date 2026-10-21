"""
Feature Reduction + HITL Experiment (All Three Locations)
==========================================================
For each unseen target location (Loc1, Loc2, Loc3):
  - Train on the other two locations (2% seed sample)
  - For k = 2..10 SHAP-selected features:
      * Measure cross-location accuracy WITHOUT HITL
      * Measure cross-location accuracy WITH HITL (800 corrections, perfect oracle)
Produces three separate graphs, one per target location.
"""

import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─── Configuration ───────────────────────────────────────────────────────────────
LOC1_FILE     = "data/rfLoc1_6dbmJamming.csv"
LOC2_FILE     = "data/rfLoc2_6dbmJamming.csv"
LOC3_FILE     = "data/rfLoc3_6dbmJamming.csv"

RANDOM_STATE  = 42
TRAIN_FRAC    = 0.02
CONF_LOWER    = 0.20
CONF_UPPER    = 0.80
MAX_HITL      = 800
FEATURE_RANGE = range(2, 11)
# ─────────────────────────────────────────────────────────────────────────────────

for f in [LOC1_FILE, LOC2_FILE, LOC3_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Dataset not found: {f}")

loc = {
    1: pd.read_csv(LOC1_FILE).dropna(),
    2: pd.read_csv(LOC2_FILE).dropna(),
    3: pd.read_csv(LOC3_FILE).dropna(),
}
print(f"Loaded  Loc1: {loc[1].shape}  Loc2: {loc[2].shape}  Loc3: {loc[3].shape}\n")

# ── Helper: run one full experiment for a given target location ──────────────────
def run_experiment(target_loc_id):
    train_ids = [i for i in [1, 2, 3] if i != target_loc_id]
    print(f"\n{'#'*65}")
    print(f"  TARGET: Location {target_loc_id}  |  "
          f"Training seed: Location {train_ids[0]} + Location {train_ids[1]}")
    print(f"{'#'*65}")

    # Seed training set
    seed_dfs = [
        loc[i].sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE)
        for i in train_ids
    ]
    train_df     = pd.concat(seed_dfs, ignore_index=True)
    X_train_full = train_df.drop(columns=["label"])
    y_train      = train_df["label"]

    # Target location: HITL pool + held-out test
    target_df = loc[target_loc_id]
    target_X  = target_df.drop(columns=["label"])
    target_y  = target_df["label"]

    X_hitl_pool, X_final_test, y_hitl_pool, y_final_test = train_test_split(
        target_X, target_y,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=target_y
    )

    print(f"  Training seed : {len(X_train_full)} samples")
    print(f"  HITL pool     : {len(X_hitl_pool)} samples")
    print(f"  Final test    : {len(X_final_test)} samples")

    # SHAP ranking from baseline 49-feature model
    print("  Computing SHAP rankings...")
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        max_features="sqrt", bootstrap=False,
        random_state=RANDOM_STATE
    )
    rf_base.fit(X_train_full, y_train)

    explainer   = shap.TreeExplainer(rf_base)
    shap_values = explainer.shap_values(X_train_full)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_df = pd.DataFrame({
        "feature": X_train_full.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    all_features_ranked = shap_df["feature"].tolist()

    # Loop over feature counts
    accs_no_hitl   = []
    accs_with_hitl = []

    print(f"\n  {'k':>3}  {'Without HITL':>14}  {'With HITL':>12}  {'Gain':>8}")
    print(f"  {'-'*45}")

    for k in FEATURE_RANGE:
        top_k = all_features_ranked[:k]

        X_train_k      = X_train_full[top_k]
        X_hitl_pool_k  = X_hitl_pool[top_k]
        X_final_test_k = X_final_test[top_k]

        # Without HITL
        rf_k = RandomForestClassifier(
            n_estimators=200, max_depth=5,
            max_features="sqrt", bootstrap=False,
            random_state=RANDOM_STATE
        )
        rf_k.fit(X_train_k, y_train)
        acc_no_hitl = accuracy_score(y_final_test, rf_k.predict(X_final_test_k))

        # Uncertain samples
        probs = rf_k.predict_proba(X_hitl_pool_k)
        uncertain_idx = [
            i for i in range(len(X_hitl_pool_k))
            if CONF_LOWER < probs[i][1] < CONF_UPPER
        ]
        if len(uncertain_idx) > MAX_HITL:
            uncertain_idx = uncertain_idx[:MAX_HITL]

        # With HITL (perfect oracle)
        corrected_X = [X_hitl_pool_k.iloc[i] for i in uncertain_idx]
        corrected_y = [int(y_hitl_pool.iloc[i]) for i in uncertain_idx]

        X_train_updated = pd.concat(
            [X_train_k,
             pd.DataFrame(corrected_X, columns=top_k)],
            ignore_index=True
        )
        y_train_updated = pd.concat(
            [y_train.reset_index(drop=True),
             pd.Series(corrected_y, dtype=int)],
            ignore_index=True
        )

        rf_hitl = RandomForestClassifier(
            n_estimators=200, max_depth=5,
            max_features="sqrt", bootstrap=False,
            random_state=RANDOM_STATE
        )
        rf_hitl.fit(X_train_updated, y_train_updated)
        acc_with_hitl = accuracy_score(y_final_test, rf_hitl.predict(X_final_test_k))

        accs_no_hitl.append(acc_no_hitl)
        accs_with_hitl.append(acc_with_hitl)

        gain = acc_with_hitl - acc_no_hitl
        print(f"  {k:>3}  {acc_no_hitl*100:>12.2f}%  {acc_with_hitl*100:>10.2f}%  "
              f"{gain*100:>+7.2f}pp")

    return accs_no_hitl, accs_with_hitl

# ── Run all three experiments ─────────────────────────────────────────────────────
all_results = {}
for target in [1, 2, 3]:
    no_hitl, with_hitl = run_experiment(target)
    all_results[target] = {"no_hitl": no_hitl, "with_hitl": with_hitl}

# ── Three separate graphs ─────────────────────────────────────────────────────────
ks = list(FEATURE_RANGE)

train_labels = {
    1: "Loc2 + Loc3",
    2: "Loc1 + Loc3",
    3: "Loc1 + Loc2",
}

for target in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(10, 5))

    no_h  = [a * 100 for a in all_results[target]["no_hitl"]]
    with_h = [a * 100 for a in all_results[target]["with_hitl"]]

    ax.plot(ks, no_h,
            marker="o", color="#e74c3c", linewidth=2,
            markersize=7, label="Without HITL")
    ax.plot(ks, with_h,
            marker="s", color="#2ecc71", linewidth=2,
            markersize=7, label="With HITL (100% expert)")

    for k, a_no, a_hi in zip(ks, no_h, with_h):
        ax.annotate(f"{a_no:.1f}%", (k, a_no),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#e74c3c")
        ax.annotate(f"{a_hi:.1f}%", (k, a_hi),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="#2ecc71")

    ax.set_xlabel("Number of SHAP-Selected Features (k)", fontsize=12)
    ax.set_ylabel("Accuracy on Test Set (%)", fontsize=12)
    ax.set_title(
        f"Feature Reduction vs. HITL — Target: Location {target}  "
        f"(Trained on {train_labels[target]})",
        fontsize=12
    )
    ax.set_xticks(ks)
    ax.set_ylim(40, 110)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = f"hitl_feature_reduction_loc{target}.png"

    out_paths = [fname]
    thesis_img = r"C:\Users\YEHIA KHALED\OneDrive\Desktop\Masters_Thesis_Template\img"
    if os.path.isdir(thesis_img):
        out_paths.append(os.path.join(thesis_img, fname))

    for p in out_paths:
        plt.savefig(p, dpi=150)
        print(f"Graph saved to {p}")

    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL k ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  OPTIMAL FEATURE COUNT ANALYSIS")
print(f"{'='*70}")

# Mean accuracy across the three locations for each k (with HITL)
mean_with_hitl = np.mean(
    [all_results[t]["with_hitl"] for t in [1, 2, 3]], axis=0
)
mean_no_hitl = np.mean(
    [all_results[t]["no_hitl"] for t in [1, 2, 3]], axis=0
)

print("\nMean accuracy across all three locations:")
print(f"  {'k':>3}  {'No HITL':>10}  {'With HITL':>10}  {'Gain vs k-1 (HITL)':>20}")
print(f"  {'-'*49}")
prev = None
for i, k in enumerate(ks):
    delta = "" if prev is None else f"{(mean_with_hitl[i] - prev)*100:+.2f}pp"
    print(f"  {k:>3}  {mean_no_hitl[i]*100:>9.2f}%  "
          f"{mean_with_hitl[i]*100:>9.2f}%  {delta:>18}")
    prev = mean_with_hitl[i]

# Best k by peak mean accuracy (with HITL)
best_idx_peak = int(np.argmax(mean_with_hitl))
best_k_peak   = ks[best_idx_peak]
best_acc_peak = mean_with_hitl[best_idx_peak]

# Saturation k: smallest k whose mean accuracy is within SAT_GAP of the peak
SAT_GAP = 0.005   # 0.5 percentage points
sat_idx = next(
    (i for i, a in enumerate(mean_with_hitl)
     if a >= best_acc_peak - SAT_GAP),
    best_idx_peak
)
sat_k   = ks[sat_idx]
sat_acc = mean_with_hitl[sat_idx]

# Best k per location (with HITL)
print("\nBest k per location (with HITL):")
for t in [1, 2, 3]:
    accs   = all_results[t]["with_hitl"]
    bi     = int(np.argmax(accs))
    print(f"  Location {t}: k = {ks[bi]}  "
          f"(accuracy = {accs[bi]*100:.2f}%)")

print(f"\n{'-'*70}")
print(f"  RECOMMENDATION")
print(f"{'-'*70}")
print(f"  Peak mean accuracy : k = {best_k_peak}  "
      f"(mean = {best_acc_peak*100:.2f}%)")
print(f"  Saturation point   : k = {sat_k}  "
      f"(mean = {sat_acc*100:.2f}%, within {SAT_GAP*100:.1f}pp of peak)")
print()
if sat_k == best_k_peak:
    print(f"  >> Use k = {best_k_peak} features. It is both the peak and the")
    print(f"     smallest count within {SAT_GAP*100:.1f}pp of the peak.")
else:
    print(f"  >> Use k = {sat_k} features. Adding features beyond this yields")
    print(f"     less than {SAT_GAP*100:.1f}pp of mean accuracy improvement,")
    print(f"     not worth the extra model complexity and inference cost.")
print(f"{'='*70}")

# ── Summary plot: mean across locations ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ks, mean_no_hitl * 100,
        marker="o", color="#e74c3c", linewidth=2,
        markersize=7, label="Without HITL (mean of 3 locations)")
ax.plot(ks, mean_with_hitl * 100,
        marker="s", color="#2ecc71", linewidth=2,
        markersize=7, label="With HITL (mean of 3 locations)")

# Mark the saturation point
ax.axvline(sat_k, color="#3498db", linestyle="--", linewidth=1.5,
           alpha=0.7, label=f"Recommended k = {sat_k}")
ax.scatter([sat_k], [sat_acc * 100], s=180, facecolors="none",
           edgecolors="#3498db", linewidths=2.5, zorder=5)

for k, a_no, a_hi in zip(ks, mean_no_hitl, mean_with_hitl):
    ax.annotate(f"{a_no*100:.1f}%", (k, a_no * 100),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color="#e74c3c")
    ax.annotate(f"{a_hi*100:.1f}%", (k, a_hi * 100),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8, color="#2ecc71")

ax.set_xlabel("Number of SHAP-Selected Features (k)", fontsize=12)
ax.set_ylabel("Mean Accuracy across 3 Locations (%)", fontsize=12)
ax.set_title("Optimal Feature Count: Mean Accuracy across All Three Locations",
             fontsize=12)
ax.set_xticks(ks)
ax.set_ylim(40, 110)
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
summary_fname = "figures/hitl_feature_reduction_summary.png"
out_paths = [summary_fname]
if os.path.isdir(thesis_img):
    out_paths.append(os.path.join(thesis_img, summary_fname))
for p in out_paths:
    plt.savefig(p, dpi=150)
    print(f"Summary graph saved to {p}")
plt.show()
