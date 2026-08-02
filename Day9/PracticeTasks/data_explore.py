import pandas as pd
from sklearn.datasets import load_iris

print("__________________________________________________")
print("              Dataset Exploartion                 ")
print("__________________________________________________")

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

print("\nRow Count:", df.shape[0], "| Attribute Count:", df.shape[1])

print("\n[Structural Head Nodes - First 5 Samples]:")
print(df.head())

print("\n[Latent Distribution Profile - Central Tendency Metrics]:")
print(df.describe())
