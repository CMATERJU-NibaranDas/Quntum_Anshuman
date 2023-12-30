# Number 1
"""
# Import necessary libraries
import numpy as np
import pandas as pd
# The q-means leverages quantum computing to calculate distances for the centroid assignment part of the k-means unsupervised learning algorithm
from qmeans.qkmeans import *

# Main code
backend = Aer.get_backend("aer_simulator_statevector") #Sets the back-end for Aer state vector simulator
X = pd.DataFrame(np.array([[1,2],[1,4],[1,0],[2,4],[3,4],[4,5],[10,3],[10,2],[10,7]])) #Sets a simple dataframe with 2-D data sets
qk_means = QuantumKMeans(backend, n_clusters=3, verbose=True) #number of clusters chosen = 3 and verbose set to true means, each computation will be shown exclusively.
qk_means.fit(X) #The qmeans model is fitted to the data-frame X for implementation
print(qk_means.labels_)

"""

# Number 2
"""
import cirq
import numpy as np

# main code
def euclidean_distance_circuit(qubit1, qubit2, ancilla):
    circuit = cirq.Circuit()

    # Apply X gates to qubit1 and qubit2
    circuit.append(cirq.X(qubit1))
    circuit.append(cirq.X(qubit2))

    # Apply controlled-Ry gate for the Euclidean distance calculation
    controlled_ry_gate = cirq.ControlledGate(cirq.ry(np.pi), num_controls=2)
    circuit.append(controlled_ry_gate(qubit1, qubit2, ancilla))

    # Apply X gates to undo the changes
    circuit.append(cirq.X(qubit1))
    circuit.append(cirq.X(qubit2))

    return circuit


# Create qubits
qubit1 = cirq.LineQubit(0)
qubit2 = cirq.LineQubit(1)
ancilla = cirq.LineQubit(2)

# Create a circuit for Euclidean distance calculation
distance_circuit = euclidean_distance_circuit(qubit1, qubit2, ancilla)

# Display the circuit
print("Euclidean Distance Circuit:")
print(distance_circuit)
"""
