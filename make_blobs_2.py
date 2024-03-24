import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Function to generate synthetic data
def generate_data(n_samples, n_features, n_clusters, cluster_std):
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=cluster_std, random_state=100)
    return data, labels

# Function to plot data points and centroids using Seaborn
def plot_results(data, assignments, centroids):
    df = pd.DataFrame({'X': data[:, 0], 'Y': data[:, 1], 'Cluster': assignments})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='X', y='Y', hue='Cluster', palette='viridis', legend='full', alpha=0.7)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red', marker='o', s=200, label='Centroids')
    plt.title('Classical K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Parameters
n_samples = 300
n_features = 2
n_clusters = 4
cluster_std = 2

# Generate synthetic data
data, true_labels = generate_data(n_samples, n_features, n_clusters, cluster_std)

# Normalize data
data -= np.min(data, axis=0)
data /= np.max(data, axis=0)

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)
assignments = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the results using Seaborn
plot_results(data, assignments, centroids)
