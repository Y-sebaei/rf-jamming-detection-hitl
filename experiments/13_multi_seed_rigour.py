"""
Multi-Seed Rigour Driver
========================
Repeats the three headline experiments across several random seeds and reports
mean +/- standard deviation, so the thesis can report variance rather than
single-seed point estimates.

Experiments:
  1. HITL cross-location recovery (target Location 1)
  2. Expert-accuracy sweep (alpha = 1.00, 0.80, 0.50) on Location 1
  3. Optimal feature count: mean post-HITL accuracy across all 3 locations,
     for k = 2..10

Run with: python RigorMultiSeed.py
"""

import pandas as pd
import numpy as np
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─── Configuration ───────────────────────────────────────────────────────────────
LOC_FILES = {
    1: "data/rfLoc1_6dbmJamming.csv",
    2: "data/rfLoc2_6dbmJamming.csv",
    3: "data/rfLoc3_6dbmJamming.csv",
}
SEEDS        = [42, 7, 123, 2024, 99]   # 5 seeds
TRAIN_FRAC   = 0.02
TOPFEATS     = 9
CONF_LOWER   = 0.20
CONF_UPPER   = 0.80
MAX_HITL     = 800
EXPERTISE    = [1.00, 0.80, 0.50]
FEATURE_RANGE = range(2, 11)
# ─────────────────────────────────────────────────────────────────────────────────

loc = {i: pd.read_csv(f).dropna() for i, f in LOC_FILES.items()}
print(f"Loaded Loc1/2/3 shapes: {[loc[i].shape for i in (1,2,3)]}")


def rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, max_features="sqrt",
        bootstrap=False, random_state=seed
    )


def shap_ranking(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.mean(np.abs(sv), axis=0)
    return (pd.DataFrame({"feature": X.columns, "imp": imp})
            .sort_values("imp", ascending=False)["feature"].tolist())


def build_split(target_id, seed):
    """Seed model on 2% of the other two locations; split target 50/50."""
    others = [i for i in (1, 2, 3) if i != target_id]
    seed_df = pd.concat(
        [loc[i].sample(frac=TRAIN_FRAC, random_state=seed) for i in others],
        ignore_index=True
    )
    Xtr, ytr = seed_df.drop(columns=["label"]), seed_df["label"]
    tX, ty = loc[target_id].drop(columns=["label"]), loc[target_id]["label"]
    Xpool, Xtest, ypool, ytest = train_test_split(
        tX, ty, test_size=0.50, random_state=seed, stratify=ty
    )
    return Xtr, ytr, Xpool, ypool, Xtest, ytest


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 + 2: HITL recovery and expertise sweep (target Location 1)
# ═══════════════════════════════════════════════════════════════════════════════
before_list, pre_list = [], []
post_by_alpha = {a: [] for a in EXPERTISE}

for seed in SEEDS:
    Xtr, ytr, Xpool, ypool, Xtest, ytest = build_split(1, seed)

    m_full = rf().fit(Xtr, ytr)
    before_list.append(accuracy_score(ytest, m_full.predict(Xtest)))

    feats = shap_ranking(m_full, Xtr)[:TOPFEATS]
    Xtr_r, Xpool_r, Xtest_r = Xtr[feats], Xpool[feats], Xtest[feats]

    m_red = rf().fit(Xtr_r, ytr)
    pre_list.append(accuracy_score(ytest, m_red.predict(Xtest_r)))

    probs = m_red.predict_proba(Xpool_r)
    unc = [i for i in range(len(Xpool_r))
           if CONF_LOWER < probs[i][1] < CONF_UPPER][:MAX_HITL]

    for alpha in EXPERTISE:
        rng = np.random.default_rng(seed)
        cy = [int(ypool.iloc[i]) if rng.random() < alpha
              else int(1 - ypool.iloc[i]) for i in unc]
        cX = pd.DataFrame([Xpool_r.iloc[i] for i in unc], columns=feats)
        Xup = pd.concat([Xtr_r, cX], ignore_index=True)
        yup = pd.concat([ytr.reset_index(drop=True),
                         pd.Series(cy, dtype=int)], ignore_index=True)
        m_up = rf().fit(Xup, yup)
        post_by_alpha[alpha].append(accuracy_score(ytest, m_up.predict(Xtest_r)))


def ms(x):
    return np.mean(x) * 100, np.std(x) * 100


print("\n" + "=" * 70)
print(f"  EXPERIMENT 1+2: HITL on Location 1 ({len(SEEDS)} seeds)")
print("=" * 70)
print(f"  Before HITL (49 feat) : {ms(before_list)[0]:5.2f}% +/- {ms(before_list)[1]:.2f}")
print(f"  Pre-HITL    (9 feat)  : {ms(pre_list)[0]:5.2f}% +/- {ms(pre_list)[1]:.2f}")
for a in EXPERTISE:
    m, s = ms(post_by_alpha[a])
    print(f"  Post-HITL alpha={a:.2f}   : {m:5.2f}% +/- {s:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Optimal feature count (mean across 3 locations, per k), multi-seed
# ═══════════════════════════════════════════════════════════════════════════════
# results[k] = list of per-(seed,location) post-HITL accuracies
post_k = {k: [] for k in FEATURE_RANGE}
nohitl_k = {k: [] for k in FEATURE_RANGE}

for seed in SEEDS:
    for target in (1, 2, 3):
        Xtr, ytr, Xpool, ypool, Xtest, ytest = build_split(target, seed)
        ranking = shap_ranking(rf().fit(Xtr, ytr), Xtr)
        for k in FEATURE_RANGE:
            fk = ranking[:k]
            Xtr_k, Xpool_k, Xtest_k = Xtr[fk], Xpool[fk], Xtest[fk]
            mk = rf().fit(Xtr_k, ytr)
            nohitl_k[k].append(accuracy_score(ytest, mk.predict(Xtest_k)))
            probs = mk.predict_proba(Xpool_k)
            unc = [i for i in range(len(Xpool_k))
                   if CONF_LOWER < probs[i][1] < CONF_UPPER][:MAX_HITL]
            cy = [int(ypool.iloc[i]) for i in unc]   # perfect oracle
            cX = pd.DataFrame([Xpool_k.iloc[i] for i in unc], columns=fk)
            Xup = pd.concat([Xtr_k, cX], ignore_index=True)
            yup = pd.concat([ytr.reset_index(drop=True),
                             pd.Series(cy, dtype=int)], ignore_index=True)
            mup = rf().fit(Xup, yup)
            post_k[k].append(accuracy_score(ytest, mup.predict(Xtest_k)))

print("\n" + "=" * 70)
print(f"  EXPERIMENT 3: Optimal k, mean across 3 locations ({len(SEEDS)} seeds)")
print("=" * 70)
print(f"  {'k':>3}  {'No HITL (mean+/-std)':>22}  {'With HITL (mean+/-std)':>24}")
means = {}
for k in FEATURE_RANGE:
    nm, ns = ms(nohitl_k[k])
    pm, ps = ms(post_k[k])
    means[k] = pm
    print(f"  {k:>3}  {nm:6.2f}% +/- {ns:4.2f}        {pm:6.2f}% +/- {ps:4.2f}")

peak_k = max(means, key=means.get)
peak = means[peak_k]
sat_k = next(k for k in FEATURE_RANGE if means[k] >= peak - 0.5)
print(f"\n  Peak mean accuracy : k = {peak_k} ({peak:.2f}%)")
print(f"  Saturation (within 0.5pp): k = {sat_k} ({means[sat_k]:.2f}%)")
print("=" * 70)
