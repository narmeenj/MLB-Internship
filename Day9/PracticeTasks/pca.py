import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

print("\n--------------------------------------------------")
print("        Principal Component Decomposition         ")
print("--------------------------------------------------")

iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print("\nOriginal Feature Coordinate Shape:", X.shape, "(4 Dimensions)")
print("Decompressed PCA Coordinate Shape:", X_transformed.shape, "(2 Dimensions)")

variance_ratios = pca.explained_variance_ratio_
print("\nEigenvalue Analysis Metrics:")
print("Variance Retained via Component Vector 1 (PC1):", "{:.2f}%".format(variance_ratios[0] * 100))
print("Variance Retained via Component Vector 2 (PC2):", "{:.2f}%".format(variance_ratios[1] * 100))

total_variance = np.sum(variance_ratios) * 100
print("\nTotal Cumulative Variance Preserved across 2D Mapping:", "{:.2f}%".format(total_variance))
print()
