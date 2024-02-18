import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'D:/QDC RS JU/processed_cleveland.csv'
df = pd.read_csv(dataset_path)

# Select numeric features for normalization and clustering
numeric_features = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]

# Normalize the numeric features
scaler = MinMaxScaler()
normalized_numeric_features = scaler.fit_transform(numeric_features)

# Apply k-means clustering to the normalized numeric features
num_clusters = 4  # Adjust the number of clusters based on your requirements
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(normalized_numeric_features)

# Get cluster centroids in the original feature space
cluster_centroids_normalized = kmeans.cluster_centers_
cluster_centroids = scaler.inverse_transform(cluster_centroids_normalized)

# Plot the cluster plots for all possible combinations of 2 numeric features
numeric_feature_combinations = [('age', 'trestbps'), ('age', 'chol'), ('age', 'thalach'), ('age', 'oldpeak'),
                                 ('trestbps', 'chol'), ('trestbps', 'thalach'), ('trestbps', 'oldpeak'),
                                 ('chol', 'thalach'), ('chol', 'oldpeak'), ('thalach', 'oldpeak')]

for combination in numeric_feature_combinations:
    feature1, feature2 = combination
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature1], df[feature2], c=df['cluster'], cmap='viridis', edgecolors='k')
    plt.scatter(cluster_centroids[:, numeric_features.columns.get_loc(feature1)],
                cluster_centroids[:, numeric_features.columns.get_loc(feature2)],
                marker='X', s=200, c='red', label='Centroids')
    plt.title(f'Cluster Plot for {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()
