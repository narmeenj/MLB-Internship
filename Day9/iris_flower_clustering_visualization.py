import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

os.makedirs("Day9", exist_ok=True)

iris = load_iris()
X = iris.data
y_true = iris.target

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, name in enumerate(iris.target_names):
    axes[0].scatter(X[y_true == i, 0], X[y_true == i, 1], label=name, color=colors[i], alpha=0.8, edgecolors="k")
axes[0].set_title("Original Topology (True Species Labels)", fontweight="bold")
axes[0].set_xlabel("Sepal Length (cm)")
axes[0].set_ylabel("Sepal Width (cm)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for i in range(optimal_k):
    axes[1].scatter(X[y_clusters == i, 0], X[y_clusters == i, 1], label=f"Cluster {i+1}", color=colors[i], alpha=0.8, edgecolors="k")
centroids = kmeans.cluster_centers_
axes[1].scatter(centroids[:, 0], centroids[:, 1], s=250, c="yellow", marker="*", edgecolors="black", label="Centroids")
axes[1].set_title("K-Means Discovered Boundaries", fontweight="bold")
axes[1].set_xlabel("Sepal Length (cm)")
axes[1].set_ylabel("Sepal Width (cm)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for i in range(optimal_k):
    axes[2].scatter(X_pca[y_clusters == i, 0], X_pca[y_clusters == i, 1], label=f"Cluster {i+1}", color=colors[i], alpha=0.8, edgecolors="k")
axes[2].set_title("PCA Maximized Variance Projection (2D Space)", fontweight="bold")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Day9/iris_clustering_analysis.png", dpi=300)
plt.savefig("iris_clustering_analysis.png", dpi=300)
plt.close()

print("[Pipeline Engine] Unified subplots exported cleanly to folder asset structures.")
