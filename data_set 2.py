# heart disease cleveland dataset
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Replace 'path/to/dataset' with the actual path where you extracted the files
dataset_path = 'D:/QDC RS JU'
df = pd.read_csv(f'{dataset_path}/processed_cleveland.csv')

# Select relevant columns for k-means (modify as needed)
features = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]

# Apply k-means clustering
num_clusters = 6  # Adjust the number of clusters based on your requirements
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Display the clustered DataFrame
print(df.head())

# Plot the clustered data
plt.scatter(features['age'], features['thalach'], c=df['cluster'], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], marker='X', s=200, c='red', label='Centroids')
plt.title('K-Means Clustering of Cleveland Clinic Heart Disease Dataset')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.legend()
plt.show()








