import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

DATA_FILE = "data/rfLoc3_6dbmJamming.csv"
RANDOM_STATE = 42

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

if "label" not in df.columns:
    raise ValueError("Missing 'label' column in dataset")

if df.isnull().any().any():
    print(f"Warning: {df.isnull().sum().sum()} NaN values found — dropping rows")
    df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=RANDOM_STATE
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_train_pred = rf.predict(X_train)

print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CV runs only on training data — avoids leaking test set information
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

print("Cross-Validation Scores (train only):", cv_scores)
print("Average CV Score:", cv_scores.mean())
