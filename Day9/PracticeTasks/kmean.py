import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

print("\n--------------------------------------------------")
print("         K-means optimization Clustering          ")
print("--------------------------------------------------")

iris = load_iris()
X = iris.data

wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print("Evaluated Configuration K =", k, "| Resulting Inertia (WCSS):", "{:.4f}".format(kmeans.inertia_))

plt.figure(figsize=(7, 4.5))
plt.plot(k_range, wcss, marker="o", linestyle="-.", color="#1f77b4", linewidth=2)
plt.title("Statistical Optimization via Elbow Diagnostics", fontsize=12, fontweight='bold')
plt.xlabel("Cluster Count Parameter (K Value)", fontsize=10)
plt.ylabel("Inertia Bound Metrics (WCSS)", fontsize=10)
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("Day9/PracticeTasks/elbow_method.png", dpi=300)
plt.close()

print("\nSuccessfully saved high-DPI artifact: elbow_method.png\n")
