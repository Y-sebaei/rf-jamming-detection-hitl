import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df = pd.read_csv("data/rfLoc1_6dbmJamming.csv")

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
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42
)

rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# TN FP
# FN TP

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

print("\nFalse Positives:", FP)
print("False Negatives:", FN)



y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)


fp_indices = np.where((y_pred_array == 1) & (y_test_array == 0))[0]


fn_indices = np.where((y_pred_array == 0) & (y_test_array == 1))[0]

print("\nNumber of False Positive Samples:", len(fp_indices))
print("Number of False Negative Samples:", len(fn_indices))



print("\n False +ve")
print(X_test.iloc[fp_indices])

print("\nFalse -ve")
print(X_test.iloc[fn_indices])


if (FP + TN) > 0:
    print("\nFalse Positive Rate:", FP / (FP + TN))

if (FN + TP) > 0:
    print("False Negative Rate:", FN / (FN + TP))