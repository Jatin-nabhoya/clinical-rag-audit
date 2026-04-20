import os
import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(df.groupby("source").size())
print(f"Total docs: {len(df)}")
print(f"Licenses: {df['license'].unique()}")
print(f"Missing files: {sum(not os.path.exists(p) for p in df['file_path'])}")