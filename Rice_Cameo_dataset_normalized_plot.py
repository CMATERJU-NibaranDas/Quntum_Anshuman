# Final code
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Excel dataset
dataset_path = 'D:/QDC RS JU/Rice_Cammeo_Osmancik.xlsx'
dataset_df = pd.read_excel(dataset_path)

# Extract numeric columns and convert to a NumPy array
numeric_columns = dataset_df.select_dtypes(include=np.number)
dataset_np_array = numeric_columns.to_numpy()

# Normalize the dataset
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dataset_np_array)

# Apply k-means clustering on normalized data
num_clusters = 2  # Setting the number of clusters to 2
kmeans_normalized = KMeans(n_clusters=num_clusters, random_state=42)
dataset_df['cluster_normalized'] = kmeans_normalized.fit_predict(normalized_data)

# Get cluster centroids in original feature space
cluster_centroids_normalized = kmeans_normalized.cluster_centers_
cluster_centroids = scaler.inverse_transform(cluster_centroids_normalized)

# Generate combinations of 3 features out of 7
feature_combinations = list(combinations(numeric_columns.columns, 3))

# Plot all combinations
for idx, combination in enumerate(feature_combinations):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    feature1, feature2, feature3 = combination

    ax.set_title(f'Cluster Plot with Features: {feature1}, {feature2}, {feature3}')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(feature3)

    for cluster in range(num_clusters):
        cluster_data = dataset_df[dataset_df['cluster_normalized'] == cluster]
        ax.scatter(cluster_data[feature1], cluster_data[feature2], cluster_data[feature3], label=f'Cluster {cluster + 1}')

    # Plot cluster centroids with darker color
    ax.scatter(cluster_centroids[:, numeric_columns.columns.get_loc(feature1)],
               cluster_centroids[:, numeric_columns.columns.get_loc(feature2)],
               cluster_centroids[:, numeric_columns.columns.get_loc(feature3)],
               marker='X', s=200, c='darkred', label='Centroids')

    # Plot axes at the positions of cluster centroids
    for centroid in cluster_centroids:
        x, y, z = centroid[numeric_columns.columns.get_loc(feature1)], centroid[numeric_columns.columns.get_loc(feature2)], centroid[numeric_columns.columns.get_loc(feature3)]
        ax.plot([x, x], [y, y], [0, z], color='black', linestyle='--', linewidth=1)
        ax.plot([x, x], [0, y], [z, z], color='black', linestyle='--', linewidth=1)
        ax.plot([0, x], [y, y], [z, z], color='black', linestyle='--', linewidth=1)

    plt.legend()
    plt.show()
    
