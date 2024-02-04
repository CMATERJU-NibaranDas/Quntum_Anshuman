# rice dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Excel dataset
dataset_path = 'D:/QDC RS JU/Rice_Cammeo_Osmancik.xlsx'
dataset_df = pd.read_excel(dataset_path)

# Extract numeric columns and convert to a NumPy array
numeric_columns = dataset_df.select_dtypes(include=np.number)
dataset_np_array = numeric_columns.to_numpy()

# Apply k-means clustering
num_clusters = 5  # Adjust the number of clusters based on your requirements
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
dataset_df['cluster'] = kmeans.fit_predict(dataset_np_array)

# Display the clustered DataFrame
print(dataset_df.head())

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot each cluster separately
for cluster in range(num_clusters):
    cluster_data = dataset_df[dataset_df['cluster'] == cluster]
    # Assuming the dataset has two features, adjust the column indices accordingly
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {cluster + 1}')

# Plot cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red', label='Centroids')

plt.title('K-Means Clustering of Rice Cammeo Osmancik Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()



