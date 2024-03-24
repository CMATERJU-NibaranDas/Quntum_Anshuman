import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister

class QuantumKMeans:
    def __init__(self, n_clusters=3, num_iterations=5, backend=Aer.get_backend('qasm_simulator'), shots=1024):
        self.n_clusters = n_clusters
        self.num_iterations = num_iterations
        self.backend = backend
        self.shots = shots

    def get_theta(self, d):
        x, y = d
        return 2 * np.arccos((x + y) / 2.0)

    def get_distance_circuit(self, theta1, theta2):
        qr = QuantumRegister(3, name="qr")
        cr = ClassicalRegister(1, name="cr")
        qc = QuantumCircuit(qr, cr)

        # Initialize qubits
        qc.h(qr[0])
        qc.h(qr[1])

        # Apply U3 gates
        qc.u(self.get_theta(theta1), np.pi, np.pi, qr[1])
        qc.u(self.get_theta(theta2), np.pi, np.pi, qr[2])

        # Controlled-swap gate
        qc.cswap(qr[0], qr[1], qr[2])

        # Measure qubit 0
        qc.measure(qr[0], cr[0])

        return qc

    def get_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def initialize_centroids(self, data):
        indices = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        return data[indices]

    def assign_to_centroids(self, data, centroids):
        assignments = []
        for point in data:
            distances = [self.get_distance(point, centroid) for centroid in centroids]
            nearest_centroid = np.argmin(distances)
            assignments.append(nearest_centroid)
        return np.array(assignments)

    def update_centroids(self, data, assignments):
        centroids = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = data[assignments == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids

    def quantum_kmeans(self, data):
        centroids = self.initialize_centroids(data)
        for _ in range(self.num_iterations):
            assignments = self.assign_to_centroids(data, centroids)
            centroids = self.update_centroids(data, assignments)
            # Convert centroids to theta values for quantum circuits
            for i in range(len(centroids) - 1):
                for j in range(i + 1, len(centroids)):
                    theta1, theta2 = centroids[i], centroids[j]
                    qc = self.get_distance_circuit(theta1, theta2)
                    job = execute(qc, self.backend, shots=self.shots)
                    counts = job.result().get_counts()
                    print(counts)  # Replace with aggregation logic for actual use
        return assignments, centroids

# Generate synthetic data
n_samples = 300
n_clusters = 4
std_deviation = 2
points, true_labels = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters, cluster_std=std_deviation, random_state=100)

# Normalize data
points -= np.min(points, axis=0)
points /= np.max(points, axis=0)

# Initialize and run Quantum K-Means
qkmeans = QuantumKMeans(n_clusters=n_clusters, num_iterations=5)
assignments, centroids = qkmeans.quantum_kmeans(points)

# Plot the results using seaborn
sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=assignments, palette='viridis', legend='full')
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red', marker='X', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Quantum K-Means Clustering')
plt.show()
