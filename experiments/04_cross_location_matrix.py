import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


loc1 = pd.read_csv("data/rfLoc1_6dbmJamming.csv")
loc2 = pd.read_csv("data/rfLoc2_6dbmJamming.csv")
loc3 = pd.read_csv("data/rfLoc3_6dbmJamming.csv")


splits = [
    {"train": [loc1, loc2], "test": loc3, "name": "Train: Loc1+Loc2 | Test: Loc3"},
    {"train": [loc1, loc3], "test": loc2, "name": "Train: Loc1+Loc3 | Test: Loc2"},
    {"train": [loc2, loc3], "test": loc1, "name": "Train: Loc2+Loc3 | Test: Loc1"},
]


rf_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
    "random_state": 42
}


for split in splits:
    
    train_df = pd.concat(split["train"], ignore_index=True)
    test_df = split["test"]
    
   
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]
    
   
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    
    
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    
    
    print(f"\n--- {split['name']} ---")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))