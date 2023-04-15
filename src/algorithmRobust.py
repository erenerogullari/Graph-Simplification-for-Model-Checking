import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from graphs import *


def graph_to_adjacency(G: nx.classes.graph.Graph) -> np.ndarray:
    """
    Converts a graph to an adjacency matrix, considering the removed nodes

    Arguments:
    G: Graph to convert

    Returns:
    A: Adjacency matrix
    """

    # Get the nodes in G as a list
    nodes = list(G.nodes)

    # Get the node with max value
    n = max(nodes)

    # Create the adjacency matrix
    A = np.zeros((n, n), dtype=int)

    # Fill the matrix
    for i in range(n):
        if i not in nodes:
            A[i, i] = -1
            continue
        for j in range(i + 1, n):
            if j in nodes and G.has_edge(i, j):
                A[i, j] = 1
                A[j, i] = 1

    return A


def adjacency_to_graph(A: np.ndarray) -> nx.classes.graph.Graph:
    """
    Converts an adjacency matrix to a graph considering the removed nodes

    Arguments:
    A: Adjacency matrix

    Returns:
    G: Resulting graph
    """

    # Create the graph
    G = nx.Graph()

    # Add the nodes
    for i in range(A.shape[0]):
        if A[i][i] != -1:
            G.add_node(i)

    # Add the edges
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] == 1:
                G.add_edge(i, j)

    return G


def remove_nodes(A: np.ndarray, S: list) -> np.ndarray:
    """
    Removes nodes from an adjacency matrix by setting the elements on their row and column to 0
    and the elements on the diagonal to -1

    Arguments:
    A: Adjacency matrix
    S: List of nodes to remove

    Returns:
    A_S: Adjacency matrix with the nodes removed
    """

    # Get the number of nodes
    n = A.shape[0]

    # Create the adjacency matrix
    A_S = np.copy(A)

    # Remove the nodes
    for i in S:
        A_S[i, :] = 0
        A_S[:, i] = 0
        A_S[i, i] = -1

    return A_S


def induced_subgraph_matrix(A: np.ndarray, S: list) -> np.ndarray:
    """
    Finds the induced subgraph of a graph

    Arguments:
    A: Adjacency matrix
    S: List of nodes

    Returns:
    A_S: Adjacency matrix of the induced subgraph
    """

    # Get the number of nodes
    n = A.shape[0]

    # Create the adjacency matrix
    A_S = np.zeros((len(S), len(S)), dtype=int)

    # Fill the matrix
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            A_S[i, j] = A[S[i], S[j]]
            A_S[j, i] = A_S[i, j]

    return A_S


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


def partition_private_neigh(A: np.ndarray, N_priv: np.ndarray, a: int, k: int, qr: int) -> dict:
    """
    Partitions (sorted asc.) the private neighbors of 'a' by the number of private neighbors s.t.
        1. the partition has at least r elements and
        2. the partition has the same number of edges as a
        3. each element in partition has the same number of private neighbors as a, and it's less than k,
                i.e. |N_priv[a][v]| = |N_priv[v][a]| <= k

    Args:
        A: Adjacency matrix of the graph
        N_priv: Matrix of the private neighbors
        a: Node to partition
        k: Max number of private neighbors
        qr: Quantifer rank
    Returns:
        partition: Partition of the private neighbors of a
    """

    # Create the partition
    partition = {}

    # Fill the partition
    for i in range(len(N_priv[a])):
        if i == a or N_priv[a][i] > k or N_priv[a][i] != N_priv[i][a] or A[i][i] == -1:
            continue
        elif N_priv[a][i] not in partition:
            partition[N_priv[a][i]] = [i]
        else:
            partition[N_priv[a][i]].append(i)

    # Sort the partition by key ascending
    partition = dict(sorted(partition.items(), key=lambda item: item[0]))

    # Remove the partitions with less than r elements
    # partition = {key: value for key, value in partition.items() if len(value) >= qr}

    # For each partition remove the values who don't have the same number of edges as a1
    for key in partition:
        partition[key] = [value for value in partition[key] if
                          len(np.where(A[a, :] == 1)[0]) == len(np.where(A[value, :] == 1)[0])]

    # Remove the partitions with less than r elements
    partition = {key: value for key, value in partition.items() if len(value) >= qr}

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

    # Number of vertices to match
    n_a = len(neighbors_a)
    n_b = len(neighbors_a)

    # Create the matching
    f = {a: b}

    # Add the tuples to the matching

    # If both vertices have each other as neighbors then remove them from the lists
    # This is symmetrical since the graph is undirected
    # Since the conditions 1-2 are already covered in the previous steps
    if b in neighbors_a:
        neighbors_a = np.delete(neighbors_a, np.where(neighbors_a == b)[0])
        n_a -= 1
        neighbors_b = np.delete(neighbors_b, np.where(neighbors_b == a)[0])
        n_b -= 1

    # Try creating fixed points first
    for neigh_a in neighbors_a:
        if neigh_a in neighbors_b:
            f[neigh_a] = neigh_a
            # Remove the matched node from the lists
            neighbors_a = np.delete(neighbors_a, np.where(neighbors_a == neigh_a)[0])
            neighbors_b = np.delete(neighbors_b, np.where(neighbors_b == neigh_a)[0])

    # Look for a and b's first degree neighbors
    for neigh_a in neighbors_a:
        for neigh_b in neighbors_b[::-1]:
            # Check if they have the same number of edges
            if np.sum(A[neigh_a, :]) == np.sum(A[neigh_b, :]):
                # Check if they have the same number of private neighbors from each other,
                # and it's less or equal than k
                if N_priv[neigh_a, neigh_b] == N_priv[neigh_b, neigh_a] and N_priv[neigh_a, neigh_b] <= k:
                    f[neigh_a] = neigh_b
                    # Remove the matched node from the list
                    neighbors_b = np.delete(neighbors_b, np.where(neighbors_b == neigh_b)[0])
                    break

    # If the matching does not start with creating fixed points then do this
    # In order to avoid unnecessary matchings see if the matching can have more fixed points
    # for v in f.keys():
    #     if f[v] in f.keys() and f[f[v]] == v:
    #         tmp = f[v]
    #         f[v] = v
    #         f[tmp] = tmp

    # Check if the matching is bijective
    if len(f.keys()) == n_a + 1 and len(f.values()) == n_b + 1:
        return f
    else:
        return {}


def satisfies_outer(A: np.ndarray, M: list) -> bool:
    """
    Checks if a list of k-tuples satisfies the outer neighborhood conditions

    Arguments:
    A: Adjacency matrix of the graph to find the k-module in
    M: List of k-tuples

    Returns:
    is_valid: True if the list of k-tuples is a k-module, False otherwise
    """

    # Check if there are 2 k-tuples in M
    if len(M) < 2:
        raise ValueError(f'The number of tuples ({len(M)}) is less than 2')

    # Neccesary functions
    t = set([item for sublist in M for item in sublist])
    outer_neighs = lambda v: set(np.where(A[v, :] == 1)[0]).difference(t)
    condition = lambda v1, v2: outer_neighs(v1) == outer_neighs(v2)

    k = len(M[0])
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            satisfies_outer = [condition(M[i][l], M[j][l]) for l in range(k)]
            if not all(satisfies_outer):
                outer1 = outer_neighs(M[i][-1])
                outer2 = outer_neighs(M[j][-1])
                nbr = np.where(A[M[j][-1], :] == 1)[0]
                return False

    return True


def find_2ktuples(A: np.ndarray, N_priv: np.ndarray, a: int, b: int, k: int) -> list:
    """
    Finds 2 k-tuples using the neighbors of a and b
    Args:
        A: Adjacency matrix
        N_priv: Matrix of private neighbors
        a: First node
        b: Second node
        k: Size of the k-tuples

    Returns:
        M: List of k-tuples
    """

    # Create an empty list for the k-tuples
    M = []

    # First match the nodes using their 1st deg neighbors
    matching = match_tuples(A, N_priv, a, b, k)

    # Pick the matchings that are not matched to themselves
    f = {}
    for i in matching.keys():
        if matching[i] != i:
            f[i] = matching[i]

    # If the matching is empty or bigger than k then return an empty list
    if len(f) == 0 or len(f) > k:
        return []

    # Add the matchings to the tuples
    M.append(tuple(f.keys()))
    M.append(tuple(f.values()))

    # If there are less than k nodes then extend the tuples
    if len(M[0]) < k:
        # Get the neighbors of the nodes
        possible_nbrs_a = list(f.keys())
        possible_nbrs_a.remove(a)
        # print('Psb nbrs: ', possible_nbrs_a)
        for nbr_a in possible_nbrs_a:
            nbr_b = f[nbr_a]
            # Match the 2 neighbors
            matching2 = match_tuples(A, N_priv, nbr_a, nbr_b, k)
            # print('Matching ', nbr_a, ' -> ', nbr_b, ': ', matching2)
            # Pick the matchings that are not matched to themselves
            for i in matching2.keys():
                # If the matching is valid, and it's not already in the tuples
                # and there is still space in the tuple then add it
                if matching2[i] != i and i not in M[0] and i not in M[1] and len(M[0]) < k:
                    # Extend f and M
                    f[i] = matching2[i]
                    M[0] += (i,)
                    M[1] += (matching2[i],)

            # If there are k elements in M
            if len(M[0]) == k:
                break

    if len(M[0]) == k:
        # and they satisfy the inner and outer nbr conditions then return M
        satisfies_in = np.array_equal(induced_subgraph_matrix(A, M[0]), induced_subgraph_matrix(A, M[1]))
        satisfies_out = satisfies_outer(A, M)
        if satisfies_in and satisfies_out:
            return M

    return []


def add_ktuple(A: np.ndarray, N_priv: np.ndarray, M: list, c: int) -> list:
    """
    Adds another k-tuple to M, if it's the 3rd one then it also reorders first 2 tuples in the k-module
    Args:
        A: Adjacency matrix
        N_priv: Matrix of private neighbors
        M: List of k-tuples
        c: New node
    Returns:
        M_: the extended k-module
    """

    # Check if there are 2 k-tuples in M
    if len(M) < 2:
        raise ValueError(f'The number of tuples ({len(M)}) is less than 2')

    # Match the 3rd node with the first 2 nodes
    num_tuple = len(M)
    k = len(M[0])
    # f = dict(zip(M[0], M[1]))  # The first matching is already in M
    mtch1 = match_tuples(A, N_priv, c, M[0][0], k)
    mtch2 = match_tuples(A, N_priv, c, M[1][0], k)

    if len(mtch1) != len(mtch2):
        print()
        print('M: ', M)
        print('c: ', c)
        print(f'Matching 1: {mtch1}')
        print(f'Matching 2: {mtch2}')
        raise ValueError(f'The number of nodes in the 2 matchings is not the same ({len(mtch1)} != {len(mtch2)})')

    # Pick the matchings that are not matched to themselves and not in the tuples
    f1, f2 = {}, {}
    keys1 = list(mtch1.keys())
    keys2 = list(mtch2.keys())
    for i in range(len(keys1)):
        key1 = keys1[i]
        key2 = keys2[i]
        if mtch1[key1] != key1 and key1 not in M[0] and key1 not in M[1]:
            f1[key1] = mtch1[key1]
        if mtch2[key2] != key2 and key2 not in M[0] and key2 not in M[1]:
            f2[key2] = mtch2[key2]

    # If the matching are empty or bigger than k then return the original k-module
    if len(f1) == 0 or len(f1) > k or len(f2) == 0 or len(f2) > k:
        return M

    # Create the extended k-module
    M_ = M.copy()
    M_.append(tuple(f1.keys()))

    # Extend the 3rd tuple if needed
    if len(M_[-1]) < k:
        # Get the neighbors of the nodes
        possible_nbrs_c = list(f1.keys())
        possible_nbrs_c.remove(c)
        for nbr_c in possible_nbrs_c:
            nbr_a = f1[nbr_c]
            nbr_b = f2[nbr_c]

            # Match the 2 neighbors
            mtch3 = match_tuples(A, N_priv, nbr_c, nbr_a, k)
            mtch4 = match_tuples(A, N_priv, nbr_c, nbr_b, k)

            # Pick the matchings that are not matched to themselves and that are not if f1
            f3, f4 = {}, {}
            keys3 = list(mtch3.keys())
            keys4 = list(mtch4.keys())
            for i in range(len(keys3)):
                key3 = keys3[i]
                key4 = keys4[i]
                if mtch3[key3] != key3 and key3 not in f1.keys():
                    f3[key3] = mtch3[key3]
                if mtch4[key4] != key4 and key4 not in f2.keys():
                    f4[key4] = mtch4[key4]

            for i in f3.keys():
                # Check if the key is already in other tuples
                is_taken = [i in M_[j] for j in range(num_tuple)]
                if not (any(is_taken)) and len(M_[-1]) < k:
                    # Extend f1, f2 and M_[2]
                    f1[i] = f3[i]
                    f2[i] = f4[i]
                    M_[-1] += (i,)

            # If there are k elements in M then break
            if len(M_[-1]) == k:
                break

    # If the tuple can't be extended then return M
    if len(M_[-1]) != k:
        return M

    # Testing for inner and outer nbr conditions
    t_a, t_b, t_c = list(M_[0]), list(M_[1]), list(M_[-1])
    T_a, T_c = induced_subgraph_matrix(A, t_a), induced_subgraph_matrix(A, t_c)
    satisfies_in = np.array_equal(T_a, T_c)
    satisfies_out = satisfies_outer(A, M_)

    if satisfies_in and satisfies_out:
        return M_
    elif satisfies_out and num_tuple == 2:
        # Try reassigning nodes to tuples
        row_indices = np.where(~(T_a == T_c).all(axis=1))[0]
        for i in row_indices:
            # Try swapping the nodes in t_a and t_c
            t_a[i], t_b[i] = t_b[i], t_a[i]

            # Check if now the induced subgraphs are equal
            if np.array_equal(induced_subgraph_matrix(A, t_a), induced_subgraph_matrix(A, t_c)):
                M_ = [tuple(t_a), tuple(t_b), tuple(t_c)]
                return M_
            else:
                # Revert the swap
                t_a[i], t_b[i] = t_b[i], t_a[i]
        # Not a valid tuple
        return M
    else:
        # Not a valid tuple
        return M


def find_kmodule(A: np.ndarray, k: int, qr: int) -> list:
    """
    Finds a k-module in a graph

    Arguments:
    A: Adjacency matrix of the graph to find the k-module in
    k: Size of the k-module
    qr: Quantifier rank

    Returns:
    M: List of k-tuples
    """

    # Find the private neighbors
    N_priv = private_neigh(A)

    # Create an empty list for the k-tuples
    M = []

    # Pick a1
    for a1 in range(A.shape[0]):
        # Skip if the node is removed
        if A[a1, a1] == -1:
            continue

        # Partition a1's private neighbors indices by the number of a1's private neighbors from that vertex
        partition = partition_private_neigh(A, N_priv, a1, k, qr)
        # print(a1, partition)

        # Iterate over partitions to find a possible k-tuple for a1
        for key in partition:
            P = partition[key]
            startId = 0
            b1, v = -1, -1

            # First try creating a 2 k-tuple
            for b1 in P:
                startId += 1

                M = find_2ktuples(A, N_priv, a1, b1, k)
                if len(M) == 2:
                    break

            # Then try adding new k-tuples to M
            for v in P[startId:]:
                # Check if v is not already in M
                is_not_used = [v not in M[i] for i in range(len(M))]
                if all(is_not_used):
                    M = add_ktuple(A, N_priv, M, v)

            if len(M) > qr:
                return M
    return []


def simplify_graph(G: nx.Graph, qr: int, displays: bool = False, k_max: int = 5) -> nx.Graph:
    """
    Simplifies a graph by removing nodes with degree less than k

    Arguments:
    G: Graph to simplify
    r: Quantifier rank
    displays: Whether to display progress
    k_max: Maximum size of the k-module to look for

    Returns:
    G2: Simplified graph
    """

    # Copy the graph
    A = graph_to_adjacency(G)

    # Lambda function for getting the number of nodes
    num_nodes = lambda A: sum([1 if A[i, i] != -1 else 0 for i in range(A.shape[0])])

    # Start checking from k=1 modules
    k = 1
    while k <= num_nodes(A) / (qr + 1) and k <= k_max:
        # Displaying progress...
        if displays:
            print(f'Looking for k={k}...')

        has_kmodules = True
        while has_kmodules:

            # Checking for k-modules...
            M = find_kmodule(A, k, qr)

            if len(M) == 0:
                # When there is no modules then stop looking for k
                has_kmodules = False
            else:
                if displays:
                    print(f'Removing {M[qr:]} from {M}')
                # Removing |M| - qr of those from G2
                to_remove = [node for subgraph in M[qr:] for node in subgraph]
                # Remove the nodes from the adjacency matrix
                A = remove_nodes(A, to_remove)

        # Increase the k for the next step
        k += 1
        # Print the number of removed vertices
    if displays:
        # print('\n--------------------------------------------------------------------------\n')
        print(f'Removed {A.shape[0] - num_nodes(A)} vertices')

    # Return the simplified graph
    G2 = adjacency_to_graph(A)
    return G2


if __name__ == "__main__":
    displays = True
    execute = [0, 1, 2]

    if 0 in execute:
        print('First graph:')
        # Create a balanced tree
        h = 3
        G = create_tree(h=h, children=[2, 3, 4])
        A = graph_to_adjacency(G)
        N_priv = private_neigh(A)

        G2 = simplify_graph(G, qr=2, displays=displays)
        find_kmodule(A, k=1, qr=2)
        print('-------------------------------------------------------------------------------------------\n')
    if 1 in execute:
        print('Second graph:')
        # Create the graph
        G = nx.Graph()
        G.add_nodes_from(range(17))
        G.add_edges_from([(0, 1), (1, 2), (1, 3),
                          (0, 4), (4, 5), (4, 6),
                          (0, 7), (7, 8), (7, 9),
                          (0, 10), (10, 11), (10, 12),
                          (3, 13), (6, 14), (9, 15), (12, 16),
                          (13, 17), (14, 17), (15, 17), (16, 17)
                          ])

        # Convert the graph to an adjacency matrix
        A = graph_to_adjacency(G)

        # Find a k-module
        G_ = simplify_graph(G, qr=2, displays=displays)
        print('-------------------------------------------------------------------------------------------\n')
    if 2 in execute:
        print('Third graph:')
        G2 = nx.Graph()
        G2.add_nodes_from(range(14))
        G2.add_edges_from([(0, 1), (1, 5), (1, 8), (1, 11), (1, 3), (2, 3),
                           (0, 4), (4, 2), (4, 8), (4, 11), (4, 6), (5, 6),
                           (0, 7), (7, 2), (7, 5), (7, 11), (7, 9), (8, 9),
                           (0, 10), (10, 2), (10, 5), (10, 8), (10, 12), (11, 12),
                           (3, 13), (6, 13), (9, 13), (12, 13)
                           ])

        # Convert the graph to an adjacency matrix
        A2 = graph_to_adjacency(G2)

        # Simplify the graph
        G2_ = simplify_graph(G2, qr=2, displays=displays)
        print('-------------------------------------------------------------------------------------------\n')
