import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the Iris dataset from UCI ML Repository
iris_data = load_iris()

# Convert the data to DataFrame
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))

# Apply PCA to reduce the dimensionality for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Visualize the data before clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df['target'], palette='Set1', legend='full')
plt.title('PCA Plot of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
df['cluster'] = kmeans.labels_

# Visualize the data after clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df['cluster'], palette='Set1', legend='full')
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(data=cluster_centers, columns=iris_data.feature_names)
print("Cluster Centers:")
print(cluster_centers_df)

# Analyze the cluster assignments
cluster_counts = df.groupby('cluster')['target'].value_counts().unstack()
print("\nCluster Assignments:")
print(cluster_counts)

# Compute silhouette score
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Compute Davies-Bouldin index
davies_bouldin = davies_bouldin_score(scaled_features, kmeans.labels_)
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

# Calculate the true class distribution
true_class_distribution = df['target'].value_counts().sort_index()

# Calculate the class distribution within each cluster
cluster_class_distribution = df.groupby('cluster')['target'].value_counts().unstack().fillna(0)

# Calculate the accuracy of each cluster
cluster_accuracies = {}

for cluster in range(len(cluster_class_distribution)):
    cluster_distribution = cluster_class_distribution.iloc[cluster]
    accuracy = max(cluster_distribution) / sum(cluster_distribution)
    cluster_accuracies[cluster] = accuracy

# Print cluster accuracies
print("\nCluster Accuracies:")
for cluster, accuracy in cluster_accuracies.items():
    print(f"Cluster {cluster}: {accuracy:.2f}")
