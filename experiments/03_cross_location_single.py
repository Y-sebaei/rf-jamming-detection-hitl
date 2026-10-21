import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load datasets
loc1 = pd.read_csv("data/rfLoc1_6dbmJamming.csv")
loc2 = pd.read_csv("data/rfLoc2_6dbmJamming.csv")
loc3 = pd.read_csv("data/rfLoc3_6dbmJamming.csv")


# Merge training locations
train_df = pd.concat([loc1, loc2], ignore_index=True)
# Test dataset — must be a location not seen during training
test_df = loc3


# Split features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


# Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42
)


# Train model
rf.fit(X_train, y_train)


# Predictions
y_pred = rf.predict(X_test)


# Evaluation
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))