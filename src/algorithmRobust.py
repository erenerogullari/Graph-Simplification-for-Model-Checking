import networkx as nx
import numpy as np

def graph_to_adjacency(G: nx.classes.graph.Graph) -> np.ndarray:
	"""
	Converts a graph to an adjacency matrix

	Arguments:
	G: Graph to convert

	Returns:
	A: Adjacency matrix
	"""

	# Get the number of nodes
	n = G.number_of_nodes()

	# Create the adjacency matrix
	A = np.zeros((n, n))

	# Fill the matrix
	for i in range(n):
		for j in range(i + 1, n):
			if G.has_edge(i, j):
				A[i, j] = 1
				A[j, i] = 1

	return A


def adjacency_to_graph(A: np.ndarray) -> nx.classes.graph.Graph:
	"""
	Converts an adjacency matrix to a graph

	Arguments:
	A: Adjacency matrix

	Returns:
	G: Resulting graph
	"""

	# Create the graph
	G = nx.Graph()

	# Add the nodes
	G.add_nodes_from(range(A.shape[0]))

	# Add the edges
	for i in range(A.shape[0]):
		for j in range(i + 1, A.shape[1]):
			if A[i, j] == 1:
				G.add_edge(i, j)

	return G


def neighbors_diff(A: np.ndarray) -> np.ndarray:
	"""
	Finds the symmetrical difference in neighbors between all pairs of nodes

	Arguments:
	A: Adjacency matrix

	Returns:
	N_delta: Matrix of the symmetrical differences in neighbors
	"""

	# Get the number of nodes
	n = A.shape[0]

	# Create the matrix
	N_delta = np.zeros((n, n))

	# Fill the matrix
	for i in range(n):
		for j in range(i + 1, n):
			neighbors_i = set(np.where(A[i, :] == 1)[0])
			neighbors_j = set(np.where(A[j, :] == 1)[0])
			N_delta[i, j] = len(neighbors_i.symmetric_difference(neighbors_j))
			N_delta[j, i] = N_delta[i, j]

	return N_delta


def find_kmodule(A: np.ndarray, k: int, min_size: int) -> list:
	"""
	Finds a k-module in a graph

	Arguments:
	A: Adjacency matrix of the graph to find the k-module in
	k: Size of the k-module
	min_size: Minimum size of the k-module

	Returns:
	M: List of k-tuples
	"""

	# Find the symmetrical difference in neighbors
	N_delta = neighbors_diff(A)

	# Find the k-module
	M = []
	for a1 in range(A.shape[0]):

		# Find the nodes whose sym diff in neighbors with a1 is even and less than 2(k-1)
		nbrs = np.where((N_delta[a1, :] % 2 == 0) & (N_delta[a1, :] < 2 * (k - 1)))[0]

		print(nbrs)


	return M


# Main
if __name__ == "__main__":
	# Create the graph
	G = nx.Graph()
	G.add_nodes_from(range(10))
	G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)])

	# Convert the graph to an adjacency matrix
	A = graph_to_adjacency(G)

	# Find a k-module
	k = 1
	min_size = 3
	M = find_kmodule(A, k, min_size)

	# Print the k-module
	print(M)