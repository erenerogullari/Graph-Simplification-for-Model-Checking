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
    A = np.zeros((n, n), dtype=int)

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


def private_neigh(A: np.ndarray) -> np.ndarray:
    """
    Finds the private neighbors of all the nodes from each node
    Args:
        A: Adjacency matrix of the graph

    Returns:
        N_priv: Private neighbors matrix
    """

    # Get the number of nodes
    n = A.shape[0]

    # Create the matrix
    N_priv = np.zeros((n, n), dtype=int)

    # Fill the matrix
    for i in range(n):
        for j in range(i + 1, n):
            neighbors_i = set(np.where(A[i, :] == 1)[0])
            neighbors_j = set(np.where(A[j, :] == 1)[0])
            # Num of rivate neigbors of i from j
            N_priv[i, j] = len(neighbors_i.difference(neighbors_j))
            # Num of rivate neigbors of j from i
            N_priv[j, i] = len(neighbors_j.difference(neighbors_i))

    return N_priv


def partition_private_neigh(A: np.ndarray, N_priv: np.ndarray, a: int, r: int) -> dict:
    """
    Partitions the private neighbors of 'a' by the number of private neighbors
    Args:
        A: Adjacency matrix of the graph
        N_priv: Matrix of the private neighbors
        a: Node to partition
        r: Quantifer rank
    Returns:
        partition: Partition of the private neighbors of a
    """

    # Create the partition
    partition = {}

    # Fill the partition
    for i in range(len(N_priv[a])):
        if N_priv[a][i] not in partition:
            partition[N_priv[a][i]] = [i]
        else:
            partition[N_priv[a][i]].append(i)

    # Sort the partition by key ascending
    partition = dict(sorted(partition.items(), key=lambda item: item[0]))

    # Remove the partitions with less than r elements
    partition = {key: value for key, value in partition.items() if len(value) >= r}

    # For each partition remove the values who don't have the same number of edges as a1
    for key in partition:
        partition[key] = [value for value in partition[key] if
                          len(np.where(A[a, :] == 1)[0]) == len(np.where(A[value, :] == 1)[0])]

    # Remove the partitions with less than r elements
    partition = {key: value for key, value in partition.items() if len(value) >= r}

    return partition


def match_tuples(A: np.ndarray, N_priv: np.ndarray, a: int, b: int, k: int) -> dict:
    """
    Finds a bijective matching between the neighbors of a and b such that
        1. they have the same number of edges and
        2. they have the same number of private neighbors from each other, and it's less or equal than k
    Args:
        A: Adjacency matrix
        N_priv: Matrix of the private neighbors
        a: First tuple
        b: Second tuple
        k: Size of the k-module

    Returns:
        f: Matching
    """

    # Check if a and b are not the same
    if a == b:
        raise ValueError("a and b must be different")

    # Get the neighbors of a and b
    neighbors_a = np.where(A[a, :] == 1)[0]
    neighbors_b = np.where(A[b, :] == 1)[0]

    # Number of neighbors to match
    n_a = len(neighbors_a)
    n_b = len(neighbors_a)

    # Create the matching
    f = {}

    # Add the tuples to the matching
    f[a] = b

    # If both vertices have each other as neighbors then remove them from the lists
    # This is symmetrical since the graph is undirected
    if b in neighbors_a:
        neighbors_a = np.delete(neighbors_a, np.where(neighbors_a == b)[0])
        neighbors_b = np.delete(neighbors_b, np.where(neighbors_b == a)[0])

    # Find a bijective matching
    for neigh_a in neighbors_a:
        for neigh_b in neighbors_b:
            # Check if they have the same number of edges
            if np.sum(A[neigh_a, :]) == np.sum(A[neigh_b, :]):
                # Check if they have the same number of private neighbors from each other,
                # and it's less or equal than k
                if N_priv[neigh_a, neigh_b] == N_priv[neigh_b, neigh_a] and N_priv[neigh_a, neigh_b] <= k:
                    f[neigh_a] = neigh_b
                    # Remove the matched node from the list
                    neighbors_b = np.delete(neighbors_b, np.where(neighbors_b == neigh_b)[0])
                    break

    # Check if the matching is bijective
    if len(f.keys()) == n_a + 1 and len(f.values()) == n_b + 1:
        return f
    else:
        return {}


def order_tuples(A: np.ndarray, nodes: list, matchings: list) -> list:
    """
    Orders 3 tuples in the k-module
    Args:
        A: Adjacency matrix
        nodes: List of nodes
        matchings: List of matchings

    Returns:
        M: Subset of the K-module
    """

    # Check if the matchins correspond to the number of nodes
    if not(len(nodes) == 3 and len(matchings) == 3):
        # Raise an error
        raise ValueError(f'The number of matchings ({len(matchings)}) '
                         f'does not correspond to the number of nodes ({len(nodes)})')

    # Create the k-module
    M = [(node,) for node in nodes]

    # Get the neighbors of the nodes
    neighbors = [np.where(A[node, :] == 1)[0] for node in nodes]

    # Iterate over the neighbors and add them into the module
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            # Get the neighbor
            neighbor = neighbors[i][j]

            # Check if the neighbor is already in any tuple in the module
            if neighbor in M[0] or neighbor in M[1] or neighbor in M[2]:
                continue

            # Check if the neighbor is the neighbor of all other nodes
            if neighbor in neighbors[(i + 1) % 3] and neighbor in neighbors[(i + 2) % 3]:
                # Check if the neighbor is matched to itself
                mtch = [matching[neighbor] == neighbor for matching in matchings]
                if all(mtch):
                    # Jump to the next neighbor since it is not in the tuple (for now)
                    continue
                elif any(mtch):
                    # This shouldn't happen
                    raise ValueError(f'{neighbor} is matched to itself in f{matchings[mtch.index(True)] + 1}')
                else:
                    # Add it to the current module
                    M[i] += (neighbor,)
            # Check if the neighbor is also the neighbor of another node
            elif neighbor in neighbors[(i + 1) % 3]:
                # Add neighbor to the module who is not a neighbor of this one
                M[(i + 2) % 3] += (neighbor,)
            elif neighbor in neighbors[(i + 2) % 3]:
                # Add neighbor to the module who is not a neighbor of this one
                M[(i + 1) % 3] += (neighbor,)
            else:
                # Add it to the current module
                M[i] += (neighbor,)

    return M


def extend_module(A: np.ndarray, M: list, v: int, matchings: list) -> list:
    """
    Extends the k-module by adding another tuple (v, ...)
    Args:
        A: Adjacency matrix
        M: K-module
        v: Node to add to the k-module
        matchings: List of matchings

    Returns:
        M: Extended k-module
    """

    # Check if the matchings length is 3
    if len(matchings) != 3:
        raise ValueError(f'The number of matchings ({len(matchings)}) is not 3')

    # Check if M has at least 3 tuples already
    if len(M) < 3:
        raise ValueError(f'The number of tuples in M ({len(M)}) is less than 3')

    # Get the neighbors of the nodes
    neighbors_a1 = np.where(A[M[0][0], :] == 1)[0]
    neighbors_b1 = np.where(A[M[1][0], :] == 1)[0]
    neighbors_c1 = np.where(A[M[2][0], :] == 1)[0]
    neighbors_v = np.where(A[v, :] == 1)[0]

    # Get the common neighbors of a1, b1, and c1
    common_neighbors = np.intersect1d(neighbors_a1, np.intersect1d(neighbors_b1, neighbors_c1))

    # Union of v's neighbors and the common neighbors minus the nodes in M
    t = [vertex for tuple_ in M for vertex in tuple_]
    possible_nodes = np.setdiff1d(np.union1d(neighbors_v, common_neighbors), t)

    # Copy M
    M_ = M.copy()

    # Add v as a new tuple to M
    M_.append((v,))

    # Iterate over v's neighbors and extend the module
    for i in range(len(possible_nodes)):
        # Get the neighbor
        node = possible_nodes[i]

        # List of booleans whether the neighbor is the neighbor of other nodes
        is_neighbor = [node in neighbors_a1, node in neighbors_b1, node in neighbors_c1]
        # Check if the neighbor is the neighbor of all other nodes
        if all(is_neighbor):
            # Check if the neighbor is matched to itself in all matching
            mtch = [matching[node] == node for matching in matchings]
            if all(mtch):
                # Jump to the next neighbor
                continue
            else:
                # Add it to the last module
                M_[-1] += (node,)
        # Check if the neighbor is not the neighbor of any other nodes
        elif not any(is_neighbor):
            # Add it to the last module
            M_[-1] += (node,)
        # Check if the neighbor is the neighbor of some nodes
        else:
            # This shouldn't happen, probably some error occurred in the previous steps
            raise ValueError(f'{node} is the neighbor of some but not all of a1, b1, and c1')

    # Check if the tuple is valid
    if len(M_[-1]) != len(M[0]):
        # Return the original module
        return M
    else:
        # Return the extended module
        return M_


def find_kmodule(A: np.ndarray, k: int, r: int) -> list:
    """
    Finds a k-module in a graph

    Arguments:
    A: Adjacency matrix of the graph to find the k-module in
    k: Size of the k-module
    r: Quantifier rank

    Returns:
    M: List of k-tuples
    """

    # Find the private neighbors
    N_priv = private_neigh(A)

    # Create an empty list for the k-tuples
    M = []

    # Pick a1
    for a1 in range(1, 2):
        # Partition a1's private neighbors indices by the number of a1's private neighbors from that vertex
        partition = partition_private_neigh(A, N_priv, a1, r)

        # Iterate over partitions to find a possible k-tuple for a1
        for key in partition:
            P = partition[key]
            startId = 0

            # Pick b1
            for id_b1, b1 in enumerate(P):
                startId += 1

                # Match a1 and b1
                f1 = match_tuples(A, N_priv, a1, b1, k)

                # If there is a matching start looking for c1
                if f1:
                    M = [tuple(f1.keys()), tuple(f1.values())]
                    # Remove the nodes who are in both tuples
                    for node in M[0]:
                        if node in M[1]:
                            M[0] = tuple([n for n in M[0] if n != node])
                            M[1] = tuple([n for n in M[1] if n != node])
                    print(f'2 tuples: {M}')
                    break

            # Pick c1
            for id_c1, c1 in enumerate(P[startId:]):
                startId += 1
                # Match a1 and b1 with c1
                f2 = match_tuples(A, N_priv, b1, c1, k)
                f3 = match_tuples(A, N_priv, a1, c1, k)

                # If there is a matching continue
                if f2 and f3:
                    print()
                    print(f'f1 ({a1}-{b1}): {f1})')
                    print(f'f2 ({b1}-{c1}): {f2})')
                    print(f'f3 ({a1}-{c1}): {f3})')
                    # Order the tuples
                    M = order_tuples(A, [a1, b1, c1], [f1, f2, f3])
                    print(f'3 Ordered tuples: {M}')
                    break

            # Find the remaining tuples
            for id_v, v in enumerate(P[startId:]):
                # Match v with a1, b1, and c1
                f4 = match_tuples(A, N_priv, a1, v, k)
                f5 = match_tuples(A, N_priv, b1, v, k)
                f6 = match_tuples(A, N_priv, c1, v, k)

                # If there is a matching continue
                if f4 and f5 and f6:
                    print()
                    print(f'f4 ({a1}-{v}): {f4})')
                    print(f'f5 ({b1}-{v}): {f5})')
                    print(f'f6 ({c1}-{v}): {f6})')
                    # Extend the module if possible
                    M = extend_module(A, M, v, [f4, f5, f6])
                    print(f'Extended module: {M}')

    return M


# Main
from plot import *
def print_array(A: np.ndarray):
    """
    Prints an array in a nice format

    Arguments:
    A: Array to print
    """

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print(A[i, j], end="  ")
        print("")


if __name__ == "__main__":
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(13))
    G.add_edges_from([(0, 1), (1, 2), (1, 3),
                        (0, 4), (4, 5), (4, 6),
                        (0, 7), (7, 8), (7, 9),
                        (0, 10), (10, 11), (10, 12)])

    # Convert the graph to an adjacency matrix
    A = graph_to_adjacency(G)
    # print(A)

    # Find a k-module
    print('First graph:')
    M = find_kmodule(A, k=4, r=2)

    ###############################################################################################
    G2 = nx.Graph()
    G2.add_nodes_from(range(13))
    G2.add_edges_from([ (0, 1), (1, 5), (1, 8), (1,11), (1, 3), (2, 3),
                        (0, 4), (4, 2), (4, 8), (4,11), (4, 6), (5, 6),
                        (0, 7), (7, 2), (7, 5), (7,11), (7, 9), (8, 9),
                        (0, 10), (10, 2), (10, 5), (10, 8), (10, 12), (11, 12)
                        ])

    # Convert the graph to an adjacency matrix
    A2 = graph_to_adjacency(G2)
    # print(A2)

    print('----------------------------------------------------')
    print('Second graph:')
    M2 = find_kmodule(A2, k=4, r=2)

    # Print the k-module
    # print(M)
