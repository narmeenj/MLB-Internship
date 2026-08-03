import pandas as pd
from sklearn.datasets import load_iris

print("--------------------------------------------------")
print("              Dataset Exploartion                 ")
print("--------------------------------------------------")

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

print("\nRow Count:", df.shape[0], "| Attribute Count:", df.shape[1])

print("\nFirst 5 Samples:")
print(df.head())

print("\nDistribution Profile Metrics:")
print(df.describe())
