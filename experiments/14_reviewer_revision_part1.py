"""
Reviewer-Revision Driver  (IC-FTAI 2026, paper 577280)
======================================================
New experiments requested by the reviewers. Uses the existing feature CSVs and
the existing pipeline settings; does not modify any existing script or dataset.

  A. Random-sampling control at equal labelling budget (uncertainty vs random),
     budgets 100/200/400/800, all three target locations.
  B. Reviewer-label-accuracy sweep alpha = 1.0 ... 0.5 in steps of 0.1.
  C. Calibration of the predicted probabilities that the uncertainty band uses
     (Brier score, 10-bin ECE, reliability table, band composition).
  D. Runtime and memory for the 49-feature and 9-feature within-location models.

Run with: python ReviewerRevision.py
"""

import pickle
import platform
import time

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

LOC_FILES = {1: "data/rfLoc1_6dbmJamming.csv",
             2: "data/rfLoc2_6dbmJamming.csv",
             3: "data/rfLoc3_6dbmJamming.csv"}
SEEDS = [42, 7, 123, 2024, 99]
TRAIN_FRAC = 0.02
TOPFEATS = 9
CONF_LOWER = 0.20
CONF_UPPER = 0.80
BUDGETS = [100, 200, 400, 800]
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

loc = {i: pd.read_csv(f).dropna() for i, f in LOC_FILES.items()}
print("Loaded Loc1/2/3 shapes:", [loc[i].shape for i in (1, 2, 3)])


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


def build_split(target_id, seed):
    others = [i for i in (1, 2, 3) if i != target_id]
    seed_df = pd.concat([loc[i].sample(frac=TRAIN_FRAC, random_state=seed)
                         for i in others], ignore_index=True)
    Xtr, ytr = seed_df.drop(columns=["label"]), seed_df["label"]
    tX, ty = loc[target_id].drop(columns=["label"]), loc[target_id]["label"]
    Xpool, Xtest, ypool, ytest = train_test_split(
        tX, ty, test_size=0.50, random_state=seed, stratify=ty)
    return Xtr, ytr, Xpool, ypool, Xtest, ytest


def retrain_eval(Xtr_r, ytr, Xpool_r, ypool, Xtest_r, ytest, idx, alpha, seed,
                 feats, want_report=False):
    rng = np.random.default_rng(seed)
    cy = [int(ypool.iloc[i]) if rng.random() < alpha else int(1 - ypool.iloc[i])
          for i in idx]
    cX = pd.DataFrame([Xpool_r.iloc[i] for i in idx], columns=feats)
    Xup = pd.concat([Xtr_r, cX], ignore_index=True)
    yup = pd.concat([ytr.reset_index(drop=True), pd.Series(cy, dtype=int)],
                    ignore_index=True)
    m = rf(seed).fit(Xup, yup)
    pred = m.predict(Xtest_r)
    acc = accuracy_score(ytest, pred)
    rep = classification_report(ytest, pred, digits=2) if want_report else None
    benign_frac = float(np.mean([int(ypool.iloc[i]) == 0 for i in idx])) if len(idx) else float("nan")
    return acc, rep, benign_frac


def ms(x):
    return np.mean(x) * 100, np.std(x) * 100


def ci95(x):
    x = np.asarray(x, dtype=float) * 100
    return 2.776 * np.std(x, ddof=1) / np.sqrt(len(x))


out = open("results/revision_part1.txt", "w", encoding="utf-8")


def say(s=""):
    print(s)
    out.write(str(s) + "\n")
    out.flush()


say("=" * 74)
say("  EXPERIMENT A: uncertainty band vs random sampling at equal budget")
say("=" * 74)

accA = {t: {"unc": {b: [] for b in BUDGETS}, "rnd": {b: [] for b in BUDGETS}}
        for t in (1, 2, 3)}
pre_acc = {t: [] for t in (1, 2, 3)}
band_size = {t: [] for t in (1, 2, 3)}
band_benign = {t: [] for t in (1, 2, 3)}
rnd_benign = {t: [] for t in (1, 2, 3)}
ref_reports = {}

for target in (1, 2, 3):
    for seed in SEEDS:
        Xtr, ytr, Xpool, ypool, Xtest, ytest = build_split(target, seed)
        m_full = rf(seed).fit(Xtr, ytr)
        feats = shap_ranking(m_full, Xtr)[:TOPFEATS]
        Xtr_r, Xpool_r, Xtest_r = Xtr[feats], Xpool[feats], Xtest[feats]
        m_red = rf(seed).fit(Xtr_r, ytr)
        pre_acc[target].append(accuracy_score(ytest, m_red.predict(Xtest_r)))

        probs = m_red.predict_proba(Xpool_r)[:, 1]
        band = [i for i in range(len(Xpool_r))
                if CONF_LOWER < probs[i] < CONF_UPPER]
        band_size[target].append(len(band))
        band_benign[target].append(np.mean([int(ypool.iloc[i]) == 0
                                            for i in band[:800]]))

        rng_sel = np.random.default_rng(1000 + seed)
        rnd_all = rng_sel.choice(len(Xpool_r), size=max(BUDGETS), replace=False)

        for b in BUDGETS:
            ref = (target == 1 and seed == 42 and b == 800)
            acc_u, rep_u, _ = retrain_eval(Xtr_r, ytr, Xpool_r, ypool, Xtest_r,
                                           ytest, band[:b], 1.0, seed, feats,
                                           want_report=ref)
            acc_r, rep_r, bf = retrain_eval(Xtr_r, ytr, Xpool_r, ypool, Xtest_r,
                                            ytest, list(rnd_all[:b]), 1.0, seed,
                                            feats, want_report=ref)
            accA[target]["unc"][b].append(acc_u)
            accA[target]["rnd"][b].append(acc_r)
            if b == 800:
                rnd_benign[target].append(bf)
            if ref:
                ref_reports["unc"] = rep_u
                ref_reports["rnd"] = rep_r

for target in (1, 2, 3):
    say("")
    say("Target Location {}   (alpha = 1.0, {} seeds)".format(target, len(SEEDS)))
    m, s = ms(pre_acc[target])
    say("  no adaptation          : {:6.2f}% +/- {:4.2f}".format(m, s))
    say("  budget      uncertainty                random        diff (pp)")
    for b in BUDGETS:
        mu, su = ms(accA[target]["unc"][b])
        mr, sr = ms(accA[target]["rnd"][b])
        say("  {:>6}   {:6.2f}% +/- {:4.2f}      {:6.2f}% +/- {:4.2f}      {:+6.2f}"
            .format(b, mu, su, mr, sr, mu - mr))
    say("  band size (mean)       : {:.1f} / {}".format(np.mean(band_size[target]),
                                                        len(loc[target]) // 2))
    say("  benign share of selected: band {:.1f}%   random {:.1f}%".format(
        np.mean(band_benign[target]) * 100, np.mean(rnd_benign[target]) * 100))

say("")
say("Paired per-seed difference (uncertainty - random), target Location 1:")
for b in BUDGETS:
    d = (np.array(accA[1]["unc"][b]) - np.array(accA[1]["rnd"][b])) * 100
    say("  budget {:>4}: mean {:+6.2f} pp   sd {:4.2f}   per seed {}   wins {}/5"
        .format(b, d.mean(), d.std(), np.round(d, 2).tolist(), int((d > 0).sum())))

say("")
say("95% CI half-widths (target Location 1, budget 800):")
say("  uncertainty {:.2f}% +/- {:.2f}".format(ms(accA[1]["unc"][800])[0],
                                              ci95(accA[1]["unc"][800])))
say("  random      {:.2f}% +/- {:.2f}".format(ms(accA[1]["rnd"][800])[0],
                                              ci95(accA[1]["rnd"][800])))

say("")
say("Reference run (seed 42, Location 1, budget 800) class-wise:")
say("--- uncertainty band ---")
say(ref_reports["unc"])
say("--- random sampling ---")
say(ref_reports["rnd"])

say("=" * 74)
say("  EXPERIMENT B: reviewer label accuracy sweep (target Location 1)")
say("=" * 74)
accB = {a: [] for a in ALPHAS}
accB_rnd = {a: [] for a in ALPHAS}
for seed in SEEDS:
    Xtr, ytr, Xpool, ypool, Xtest, ytest = build_split(1, seed)
    m_full = rf(seed).fit(Xtr, ytr)
    feats = shap_ranking(m_full, Xtr)[:TOPFEATS]
    Xtr_r, Xpool_r, Xtest_r = Xtr[feats], Xpool[feats], Xtest[feats]
    m_red = rf(seed).fit(Xtr_r, ytr)
    probs = m_red.predict_proba(Xpool_r)[:, 1]
    band = [i for i in range(len(Xpool_r))
            if CONF_LOWER < probs[i] < CONF_UPPER][:800]
    rng_sel = np.random.default_rng(1000 + seed)
    rnd = list(rng_sel.choice(len(Xpool_r), size=800, replace=False))
    for a in ALPHAS:
        accB[a].append(retrain_eval(Xtr_r, ytr, Xpool_r, ypool, Xtest_r, ytest,
                                    band, a, seed, feats)[0])
        accB_rnd[a].append(retrain_eval(Xtr_r, ytr, Xpool_r, ypool, Xtest_r,
                                        ytest, rnd, a, seed, feats)[0])
say("  alpha        uncertainty                random")
for a in ALPHAS:
    mu, su = ms(accB[a])
    mr, sr = ms(accB_rnd[a])
    say("  {:>5.1f}    {:6.2f}% +/- {:4.2f}      {:6.2f}% +/- {:4.2f}"
        .format(a, mu, su, mr, sr))

say("=" * 74)
say("  EXPERIMENT C: calibration of the probabilities used by the band")
say("=" * 74)


def calib_block(name, p, y):
    y = np.asarray(y, dtype=int)
    brier = brier_score_loss(y, p)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    rows = []
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        conf, freq, n = p[mask].mean(), y[mask].mean(), int(mask.sum())
        ece += n / len(p) * abs(conf - freq)
        rows.append(("[{:.1f},{:.1f})".format(bins[b], bins[b + 1]), n, conf, freq))
    say("")
    say("{}:  Brier = {:.4f}   10-bin ECE = {:.4f}   n = {}".format(
        name, brier, ece, len(p)))
    say("           bin      n  mean pred  obs. freq")
    for r in rows:
        say("  {:>12} {:>6} {:>10.3f} {:>10.3f}".format(*r))
    in_band = (p > CONF_LOWER) & (p < CONF_UPPER)
    if in_band.any():
        say("  inside band {}-{}: n = {} ({:.1f}%)   mean pred = {:.3f}   "
            "observed jamming frequency = {:.3f}".format(
                CONF_LOWER, CONF_UPPER, int(in_band.sum()), in_band.mean() * 100,
                p[in_band].mean(), y[in_band].mean()))


Xtr, ytr, Xpool, ypool, Xtest, ytest = build_split(1, 42)
m_full = rf(42).fit(Xtr, ytr)
feats9 = shap_ranking(m_full, Xtr)[:TOPFEATS]
m_red = rf(42).fit(Xtr[feats9], ytr)
calib_block("Cross-location seed model on the Location 1 review pool (seed 42)",
            m_red.predict_proba(Xpool[feats9])[:, 1], ypool)

d1 = loc[1]
X1, y1 = d1.drop(columns=["label"]), d1["label"]
X1tr, X1te, y1tr, y1te = train_test_split(X1, y1, test_size=0.30,
                                          random_state=42, stratify=y1)
m1 = rf(42).fit(X1tr, y1tr)
calib_block("Within-location model on the Location 1 test split (49 features)",
            m1.predict_proba(X1te)[:, 1], y1te)

say("=" * 74)
say("  EXPERIMENT D: runtime and memory, 49 vs 9 features (within location)")
say("=" * 74)
say("  platform: {} | {} {} | python {}".format(
    platform.processor() or platform.machine(), platform.system(),
    platform.release(), platform.python_version()))

cmp_csv = pd.read_csv("results/feature_importance_comparison.csv")
feats_within9 = cmp_csv.sort_values("shap_rank").head(9)["feature"].tolist()
say("  9-feature set (within-location SHAP ranking): {}".format(feats_within9))


def bench(Xtr_, ytr_, Xte_, yte_, label, repeats=5):
    tt, it = [], []
    pred = None
    m = None
    for _ in range(repeats):
        m = rf(42)
        t0 = time.perf_counter()
        m.fit(Xtr_, ytr_)
        tt.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pred = m.predict(Xte_)
        it.append(time.perf_counter() - t0)
    acc = accuracy_score(yte_, pred)
    nodes = int(sum(e.tree_.node_count for e in m.estimators_))
    size = len(pickle.dumps(m))
    say("")
    say("  " + label)
    say("    accuracy            : {:.2f}%".format(acc * 100))
    say("    train time (s)      : {:.3f} +/- {:.3f}".format(np.mean(tt), np.std(tt)))
    say("    inference time (s)  : {:.4f} +/- {:.4f}   ({:.1f} us/window)".format(
        np.mean(it), np.std(it), np.mean(it) / len(Xte_) * 1e6))
    say("    total tree nodes    : {}".format(nodes))
    say("    pickled model (KiB) : {:.1f}".format(size / 1024))


bench(X1tr, y1tr, X1te, y1te, "49 features")
bench(X1tr[feats_within9], y1tr, X1te[feats_within9], y1te, "9 features")

say("")
say("Done.")
out.close()
