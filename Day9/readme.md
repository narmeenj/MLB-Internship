# Day 9 Mini Project: Iris Flower Clustering & Visualization

This project explores Unsupervised Machine Learning techniques by applying **K-Means Clustering** and **Principal Component Analysis (PCA)** to the classic Scikit-learn Iris Dataset.

## Core Concepts Explained

### 1. What is Clustering?
Clustering is an unsupervised machine learning task that automatically divides unlabelled data points into distinct groups or "clusters". The objective is to maximize similarity within a cluster (high intra-cluster similarity) and minimize similarity between different clusters (low inter-cluster similarity). 

### 2. What is PCA?
Principal Component Analysis (PCA) is a dimensionality reduction technique. It transforms high-dimensional features into a smaller set of orthogonal axes called **Principal Components**. These components capture the maximum variance in the data, making high-dimensional trends easily visualizable in a 2D or 3D space without losing vital information.

## Project Methodology & Implementation

### Determining the Optimal K Value
The optimal number of clusters was determined using the **Elbow Method**. 
* We calculated the **Within-Cluster Sum of Squares (WCSS)** for range $K = 1$ to $10$.
* Plotting WCSS against the number of clusters shows a distinct bend or "elbow" point at **$K = 3$**. This mathematical inflection confirms that partitioning the data into three separate groups optimizes cluster compactness.

---

## Visual Insights & Observations

### 1. How many clusters were formed?
Exactly **3 clusters** were generated using the K-Means algorithm, mirroring our findings from the Elbow plot.

### 2. Did the clusters represent the flower species well?
* **Setosa** was separated into its own flawless cluster perfectly, showing complete alignment between unsupervised clustering and true biological target labels.
* **Versicolor** and **Virginica** showed minor, overlapping boundaries. This occurs naturally because their physical measurements closely blend in raw feature space.

### 3. How did PCA help in visualization?
The original Iris dataset contains **4 dimensions** (Sepal/Petal lengths and widths), which is impossible to chart natively. By compressing these 4 dimensions into **2 Principal Components**, PCA captured roughly $97.7\%$ of data variance. 
* Looking at the standard features plot (Sepal Length vs Sepal Width), the clusters significantly blend together.
* Looking at the **PCA-Transformed plot**, the data points stretch out cleanly, maximizing separation gaps and proving that PCA makes unsupervised patterns visually distinct and easier to analyze.
