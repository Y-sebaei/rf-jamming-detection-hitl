"""Compute 5-fold CV (training fold) and the feature correlation matrix for the defense."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/rfLoc1_6dbmJamming.csv").dropna()
X, y = df.drop(columns=["label"]), df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=5, max_features="sqrt",
                            bootstrap=False, random_state=42)
rf.fit(Xtr, ytr)
cv = cross_val_score(rf, Xtr, ytr, cv=5, scoring="accuracy")
print(f"CV_MEAN={cv.mean()*100:.2f} CV_STD={cv.std()*100:.2f} folds={[round(c*100,2) for c in cv]}")

# Correlation among the 9 SHAP-top features (and total_gain_db family)
top9 = ["total_gain_db_mean","total_gain_db_q25","avgpwr_db_max","rssi_std",
        "max_magnitude_max","max_magnitude_std","total_gain_db_q50",
        "base_pwr_db_q75","total_gain_db_min"]
top9 = [c for c in top9 if c in X.columns]
corr = X[top9].corr().abs()
# Mean off-diagonal correlation
off = corr.where(~np.eye(len(corr), dtype=bool)).stack()
print(f"MEAN_ABS_OFFDIAG_CORR_top9={off.mean():.3f}")
# total_gain_db family correlation
tg = [c for c in X.columns if c.startswith("total_gain_db")]
tgc = X[tg].corr().abs()
offtg = tgc.where(~np.eye(len(tgc), dtype=bool)).stack()
print(f"TOTAL_GAIN_FAMILY_MEAN_CORR={offtg.mean():.3f} (n={len(tg)} features)")

# Heatmap of full 49x49 (or top families)
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(X[top9].corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(top9))); ax.set_xticklabels(top9, rotation=90, fontsize=6)
ax.set_yticks(range(len(top9))); ax.set_yticklabels(top9, fontsize=6)
ax.set_title("Correlation among top-9 SHAP features (Location 1)", fontsize=10)
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
import os
for p in ["figures/feature_correlation_top9.png",
          r"C:\Users\YEHIA KHALED\OneDrive\Desktop\Masters_Thesis_Template\img\feature_correlation_top9.png"]:
    if p.startswith("feature") or os.path.isdir(os.path.dirname(p)):
        plt.savefig(p, dpi=150)
print("saved feature_correlation_top9.png")
