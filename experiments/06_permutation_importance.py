import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


topfeats = 9
RANDOM_STATE = 42


df = pd.read_csv("data/rfLoc1_6dbmJamming.csv")

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset loaded:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

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

print(f"Baseline accuracy (all features): {baseline_acc:.4f}")



perm_result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm_result.importances_mean
}).sort_values(by="importance", ascending=False)


top_features = perm_importance_df.head(topfeats)["feature"].tolist()

print("\nTop Permutation-selected features:")
for f in top_features:
    print(" -", f)




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