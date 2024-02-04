#Q k-means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qmeans.qkmeans import QuantumKMeans
from qiskit import Aer

# URL of the Iris dataset
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Column names for the Iris dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Use pandas to read the dataset from the URL
iris_df = pd.read_csv(dataset_url, header=None, names=column_names)

# Extract numeric columns and convert to a NumPy array
numeric_columns = iris_df.select_dtypes(include=np.number)
iris_np_array = numeric_columns.to_numpy()

# Using q-means
backend = Aer.get_backend("aer_simulator_statevector")
qk_means = QuantumKMeans(backend, n_clusters=4, verbose=True)
qk_means.fit(iris_np_array)

# Get cluster labels and centroids
cluster_labels = qk_means.labels_
centroids = qk_means.cluster_centers_

# Add cluster labels to the original DataFrame for analysis
iris_df['cluster'] = cluster_labels

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot each cluster separately
for cluster in range(3):
    cluster_data = iris_df[iris_df['cluster'] == cluster]
    plt.scatter(cluster_data['sepal_length'], cluster_data['sepal_width'], label=f'Cluster {cluster + 1}')

# Plot cluster centroids
plt.scatter(centroids[0], centroids[1], marker='X', s=200, c='red', label='Centroids')

plt.title('Quantum K-Means Clustering of Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Classical k-means

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# URL of the Iris dataset
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Column names for the Iris dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Use pandas to read the dataset from the URL
iris_df = pd.read_csv(dataset_url, header=None, names=column_names)

# Extract numeric columns and convert to a NumPy array
numeric_columns = iris_df.select_dtypes(include=np.number)
iris_np_array = numeric_columns.to_numpy()


# Classical k-means analysis
# Number of clusters (you can adjust this based on your requirements)
num_clusters = 4

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(iris_np_array)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add cluster labels to the original DataFrame for analysis
iris_df['cluster'] = cluster_labels

# Display the clustered DataFrame
print(iris_df)

# Plot the clusters based on the first two features (you can modify for other features)
plt.scatter(iris_np_array[:, 0], iris_np_array[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='r', label='Centroids')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
