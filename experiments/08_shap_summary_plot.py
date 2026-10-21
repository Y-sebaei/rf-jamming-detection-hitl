import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/rfLoc3_6dbmJamming.csv")

X = df.drop(columns=["label"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="sqrt",
    bootstrap=False,
    random_state=42
)
rf.fit(X_train, y_train)


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)


if isinstance(shap_values, list):
    shap_vals = shap_values[1]   # 1 for jamming
else:
    shap_vals = shap_values


mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": mean_abs_shap
}).sort_values(by="importance", ascending=False)


feature_counts = range(2, 21)
accuracies = []

for k in feature_counts:
    top_features = importance_df.head(k)["feature"].tolist()

    rf_k = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features="sqrt",
        bootstrap=False,
        random_state=42
    )

    rf_k.fit(X_train[top_features], y_train)
    y_pred = rf_k.predict(X_test[top_features])

    accuracies.append(accuracy_score(y_test, y_pred))


plt.figure()
plt.plot(feature_counts, accuracies, marker="o")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Features")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/accuracy_vs_features.png", dpi=150)
print("Plot saved to accuracy_vs_features.png")
plt.show()
