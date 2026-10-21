import pandas as pd

benign = pd.read_csv("featuresLoc3.csv")
jamming = pd.read_csv("features6dbmJamming.csv")

final_df = pd.concat([benign, jamming], ignore_index=True)

final_df.to_csv("data/rfLoc3_6dbmJamming.csv", index=False)

print("Final dataset created")
print("Shape:", final_df.shape)
print(final_df["label"].value_counts())
