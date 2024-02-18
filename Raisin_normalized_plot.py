import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import itertools
from mpl_toolkits.mplot3d import Axes3D


your_dataset_path = 'D:/QDC RS JU/Raisin_Dataset.xlsx'
your_dataset_df = pd.read_excel(your_dataset_path)

# Drop non-numeric columns
your_numeric_df = your_dataset_df.select_dtypes(include=np.number)

# Normalizing each column
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(your_numeric_df)

normalized_df = pd.DataFrame(normalized_data, columns=your_numeric_df.columns)

# Apply K-means clustering with 2 clusters
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Get all possible combinations of 3 features
feature_combinations = list(itertools.combinations(normalized_df.columns, 3))

# Plotting all combinations
for features in feature_combinations:
    # Fit K-means clustering to the selected features
    kmeans.fit(normalized_df[list(features)])
    
    # Get cluster labels and centroids
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Plotting the clusters in 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster separately
    for cluster in range(num_clusters):
        cluster_data = normalized_df[cluster_labels == cluster]
        ax.scatter(cluster_data[features[0]], cluster_data[features[1]], cluster_data[features[2]], label=f'Cluster {cluster + 1}')
    
    # Plot cluster centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='X', s=200, c='red', label='Centroids')
    
    ax.set_title(f'K-Means Clustering of Raisin Dataset\nFeatures: {", ".join(features)} (Normalized)')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend()
    
    plt.show()
