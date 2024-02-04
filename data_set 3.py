#raisin dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your Excel dataset
your_dataset_path = 'D:/QDC RS JU/Raisin_Dataset.xlsx'
your_dataset_df = pd.read_excel(your_dataset_path)

# Extract numeric columns and convert to a NumPy array
numeric_columns_your = your_dataset_df.select_dtypes(include=np.number)
your_np_array = numeric_columns_your.to_numpy()

# Apply k-means clustering
num_clusters = 3  # Adjust the number of clusters based on your requirements
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
your_dataset_df['cluster'] = kmeans.fit_predict(your_np_array)

# Display the clustered DataFrame
print(your_dataset_df.head())

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot each cluster separately
for cluster in range(num_clusters):
    cluster_data = your_dataset_df[your_dataset_df['cluster'] == cluster]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {cluster + 1}')

# Plot cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red', label='Centroids')

plt.title('K-Means Clustering of Raisin Dataset')
plt.xlabel('Your Feature 1')
plt.ylabel('Your Feature 2')
plt.legend()
plt.show()






