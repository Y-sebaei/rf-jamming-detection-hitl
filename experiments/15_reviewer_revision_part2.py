"""
Reviewer-Revision Driver, part 2  (IC-FTAI 2026, paper 577280)
==============================================================
  E. Magnitude of window-level-split optimism: random stratified 70/30 split
     versus a contiguous (order-preserving) 70/30 split of each class block,
     which keeps temporally adjacent windows on the same side of the split.
  F. Adaptation strategies that need no human labels, for context:
       F1 no adaptation
       F2 per-location feature standardisation (unsupervised domain adaptation)
       F3 self-training on the model's own confident pool predictions
       F4 random 800 reviewer labels (reference, from part 1)

Run with: python ReviewerRevision2.py
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

LOC_FILES = {1: "data/rfLoc1_6dbmJamming.csv",
             2: "data/rfLoc2_6dbmJamming.csv",
             3: "data/rfLoc3_6dbmJamming.csv"}
SEEDS = [42, 7, 123, 2024, 99]
TRAIN_FRAC = 0.02
TOPFEATS = 9
PSEUDO_LOW = 0.10
PSEUDO_HIGH = 0.90

loc = {i: pd.read_csv(f).dropna() for i, f in LOC_FILES.items()}

out = open("results/revision_part2.txt", "w", encoding="utf-8")


def say(s=""):
    print(s)
    out.write(str(s) + "\n")
    out.flush()


def rf(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=5,
                                  max_features="sqrt", bootstrap=False,
                                  random_state=seed)


def shap_ranking(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.mean(np.abs(sv), axis=0)
    return (pd.DataFrame({"feature": X.columns, "imp": imp})
            .sort_values("imp", ascending=False)["feature"].tolist())


def ms(x):
    return np.mean(x) * 100, np.std(x) * 100


say("=" * 74)
say("  EXPERIMENT E: random vs contiguous (order-preserving) split")
say("=" * 74)
say("  A contiguous split keeps temporally adjacent windows together, so the")
say("  gap to the random split bounds the optimism of window-level splitting.")
say("  It also introduces any drift within a recording, so it is an upper bound.")

cmp_csv = pd.read_csv("results/feature_importance_comparison.csv")
feats_within9 = cmp_csv.sort_values("shap_rank").head(9)["feature"].tolist()

for target in (1, 2, 3):
    d = loc[target].reset_index(drop=True)
    X, y = d.drop(columns=["label"]), d["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=42, stratify=y)
    m = rf(42).fit(Xtr, ytr)
    acc_rand49 = accuracy_score(yte, m.predict(Xte))
    m9 = rf(42).fit(Xtr[feats_within9], ytr)
    acc_rand9 = accuracy_score(yte, m9.predict(Xte[feats_within9]))

    # contiguous: first 70% of each class block (in file order) to train
    tr_idx, te_idx = [], []
    for cls in (0, 1):
        blk = d.index[d["label"] == cls].tolist()
        cut = int(round(0.70 * len(blk)))
        tr_idx += blk[:cut]
        te_idx += blk[cut:]
    Xtrc, ytrc = X.loc[tr_idx], y.loc[tr_idx]
    Xtec, ytec = X.loc[te_idx], y.loc[te_idx]
    mc = rf(42).fit(Xtrc, ytrc)
    acc_cont49 = accuracy_score(ytec, mc.predict(Xtec))
    mc9 = rf(42).fit(Xtrc[feats_within9], ytrc)
    acc_cont9 = accuracy_score(ytec, mc9.predict(Xtec[feats_within9]))

    say("")
    say("Location {}  (train {} / test {})".format(target, len(tr_idx), len(te_idx)))
    say("  49 features: random {:6.2f}%   contiguous {:6.2f}%   gap {:+6.2f} pp"
        .format(acc_rand49 * 100, acc_cont49 * 100,
                (acc_cont49 - acc_rand49) * 100))
    say("   9 features: random {:6.2f}%   contiguous {:6.2f}%   gap {:+6.2f} pp"
        .format(acc_rand9 * 100, acc_cont9 * 100,
                (acc_cont9 - acc_rand9) * 100))
    if target == 1:
        say("  contiguous-split class-wise report (49 features):")
        say(classification_report(ytec, mc.predict(Xtec), digits=2))

say("=" * 74)
say("  EXPERIMENT F: label-free adaptation baselines (target Location 1)")
say("=" * 74)

res = {k: [] for k in ("none", "stdz", "self", "rnd800")}
pseudo_n, pseudo_acc = [], []

for seed in SEEDS:
    others = [i for i in (1, 2, 3) if i != 1]
    seed_parts = [loc[i].sample(frac=TRAIN_FRAC, random_state=seed) for i in others]
    seed_df = pd.concat(seed_parts, ignore_index=True)
    Xtr, ytr = seed_df.drop(columns=["label"]), seed_df["label"]
    tX, ty = loc[1].drop(columns=["label"]), loc[1]["label"]
    Xpool, Xtest, ypool, ytest = train_test_split(tX, ty, test_size=0.50,
                                                  random_state=seed, stratify=ty)

    feats = shap_ranking(rf(seed).fit(Xtr, ytr), Xtr)[:TOPFEATS]
    Xtr_r, Xpool_r, Xtest_r = Xtr[feats], Xpool[feats], Xtest[feats]
    m0 = rf(seed).fit(Xtr_r, ytr)
    res["none"].append(accuracy_score(ytest, m0.predict(Xtest_r)))

    # F2: standardise each location with its own unlabelled statistics
    zparts = []
    for part in seed_parts:
        p = part[feats]
        zparts.append((p - p.mean()) / p.std(ddof=0).replace(0, 1))
    Xtr_z = pd.concat(zparts, ignore_index=True)
    tgt = loc[1][feats]
    mu, sd = tgt.mean(), tgt.std(ddof=0).replace(0, 1)
    Xtest_z = (Xtest_r - mu) / sd
    mz = rf(seed).fit(Xtr_z, ytr)
    res["stdz"].append(accuracy_score(ytest, mz.predict(Xtest_z)))

    # F3: self-training on confident pool predictions, no human labels
    p_pool = m0.predict_proba(Xpool_r)[:, 1]
    conf = np.where((p_pool < PSEUDO_LOW) | (p_pool > PSEUDO_HIGH))[0]
    py = (p_pool[conf] > 0.5).astype(int)
    pseudo_n.append(len(conf))
    pseudo_acc.append(accuracy_score(ypool.iloc[conf].astype(int), py)
                      if len(conf) else np.nan)
    Xup = pd.concat([Xtr_r, pd.DataFrame(Xpool_r.iloc[conf].values, columns=feats)],
                    ignore_index=True)
    yup = pd.concat([ytr.reset_index(drop=True), pd.Series(py, dtype=int)],
                    ignore_index=True)
    ms_ = rf(seed).fit(Xup, yup)
    res["self"].append(accuracy_score(ytest, ms_.predict(Xtest_r)))

    # F4: 800 random reviewer labels, alpha = 1.0 (same selection as part 1)
    rng_sel = np.random.default_rng(1000 + seed)
    ridx = list(rng_sel.choice(len(Xpool_r), size=800, replace=False))
    cX = pd.DataFrame(Xpool_r.iloc[ridx].values, columns=feats)
    cy = ypool.iloc[ridx].astype(int).tolist()
    Xup = pd.concat([Xtr_r, cX], ignore_index=True)
    yup = pd.concat([ytr.reset_index(drop=True), pd.Series(cy, dtype=int)],
                    ignore_index=True)
    mr = rf(seed).fit(Xup, yup)
    res["rnd800"].append(accuracy_score(ytest, mr.predict(Xtest_r)))

labels = {"none": "no adaptation (9 feat.)",
          "stdz": "per-location standardisation (no labels)",
          "self": "self-training, confident pseudo-labels (no labels)",
          "rnd800": "800 random reviewer labels (alpha = 1.0)"}
say("")
for k in ("none", "stdz", "self", "rnd800"):
    m, s = ms(res[k])
    say("  {:<52} {:6.2f}% +/- {:4.2f}".format(labels[k], m, s))
say("")
say("  pseudo-labels added (mean): {:.0f}   of which correct: {:.1f}%".format(
    np.mean(pseudo_n), np.nanmean(pseudo_acc) * 100))

say("")
say("Done.")
out.close()
