import networkx as nx
from networkx.algorithms import isomorphism
import itertools
from IPython.display import display
import random


def is_kmodule(G: nx.classes.graph.Graph, nodes1: list, nodes2: list, testing: bool = False) -> bool:
    """
    Checks if 2 subgraphs are k-modules

    Arguments:
    G: The graph to check
    nodes1: First nodes list
    nodes2: Second nodes list
    testing: Whether to print steps - used for debugging

    Return:
    bool: Whether the 2 subgraphs are k-modules
    """

    # All nodes and edges
    nodes = G.nodes
    edges = G.edges

    # Induced subgraphs
    G1 = nx.subgraph(G, nodes1)
    G2 = nx.subgraph(G, nodes2)

    # Check if the induced graphs have the same number of vertices
    if len(nodes1) != len(nodes2):
        if testing: print('The modules are not of same size of vertices!')
        return False

    # Check if the induced graphs are non-empty
    if len(nodes1) == 0:
        if testing: print('The modules are empty!')
        return False

    # Check if the modules are disjoint
    if len([n for n in nodes1 if n in nodes2]) > 0:
        if testing: print('The modules are not disjoint!')
        return False

    # Check if the induced graphs have the same number of edges
    if G1.number_of_edges() != G2.number_of_edges():
        if testing: print('The modules are not of same size of edges!')
        return False

    # Obtain all possible isomorphisms between 2 subgraphs
    GM = isomorphism.GraphMatcher(G1, G2)
    isomorphs = list(GM.isomorphisms_iter())

    # If there is no isomorphism
    if len(isomorphs) == 0:
        if testing: print('The modules are not isomorphic!')
        return False

    # Iterate over all isomorphisms to find a matching that satisfies k-modules
    for isomorph in isomorphs:
        satisfies_inner = True
        satisfies_outer = True

        # Reordering the modules according to the isomorphism...
        nodes1 = list(isomorph.keys())
        nodes2 = list(isomorph.values())

        # Testing the neigboorhood properties...
        # Iterating over the nodes of the first subgraph...
        for id1, node1 in enumerate(nodes1):

            # Checking the neighbourhood properties between the 2 modules...
            inner_nbrs1 = [n for n in G.neighbors(node1) if n in nodes2]

            for nbr in inner_nbrs1:
                # If there is an edge (a_i, b_j) then there should be (a_j, b_i)
                id2 = nodes2.index(nbr)
                if not ((nodes1[id2], nodes2[id1]) in edges):
                    if testing:
                        print('Not satisfying the inner neighbourhood properties')
                        print('Matching: ', isomorph)
                        print('There is', (node1, nbr))
                        print('But no', (nodes1[id2], nodes2[id1]))

                    satisfies_inner = False

            # Checking the neighbourhood properties outside the 2 modules...
            node2 = nodes2[id1]

            # Outer neighbours
            outer_nbrs1 = [n for n in G.neighbors(node1) if n not in nodes1 and n not in nodes2]
            outer_nbrs2 = [n for n in G.neighbors(node2) if n not in nodes1 and n not in nodes2]

            if outer_nbrs1 != outer_nbrs2:
                if testing:
                    print('Not satisfying the outer neighbourhood properties:')
                    print('Matching: ', isomorph)
                    print('Outer neighbours of', node1, ':', outer_nbrs1)
                    print('Outer neighbours of', node2, ':', outer_nbrs2)
                satisfies_outer = False

        # If both neighbourhood properties are satisfied then the subgraphs are k-modules
        if satisfies_inner and satisfies_outer:
            if testing:
                print('Matching: ', isomorph)
            return True

    # Return False if there is no isomorphism satisfying the neighbourhood properties
    return False


def are_kmodule(G: nx.classes.graph.Graph, M: list) -> bool:
    """
    Checks if a set of subgraphs are pairwise k-modules

    Arguments:
    G: The graph to check
    M: The set of subgraphs (only nodes as a list)

    Return:
    bool: Whether all subgraphs are pairwise k-modules
    """

    # Check if the list is empty
    if len(M) == 0: return False

    # Testing for all unique pairs in M if they are k-modules
    for m1 in range(len(M)):
        for m2 in range(m1 + 1, len(M)):

            # If one such pair is found then return False
            if not (is_kmodule(G, M[m1], M[m2])): return False

    # Return True if there is no pair that aren't k-modules
    return True


def partition_subgraphs(G: nx.classes.graph.Graph, subgraphs: list) -> list:
    """
    Partitions a list of subgraphs wrt their number of edges

    Arguments:
    G: The graph to check
    subgraphs: A list of subgraphs in G

    Return:
    subgraphs_partitioned: A list of lists where each list have subgraphs of same number of edges
    """

    # Create a dictionary to hold the subgraphs grouped by the number of edges
    subgraphs_by_edges = {}

    # Iterate over all subgraphs and group them by the number of edges
    for subgraph in subgraphs:
        num_edges = G.subgraph(subgraph).number_of_edges()  # get the number of edges in the subgraph
        if num_edges not in subgraphs_by_edges:
            subgraphs_by_edges[num_edges] = []  # create a new list for this number of edges if it doesn't exist
        subgraphs_by_edges[num_edges].append(subgraph)  # add the subgraph to the appropriate list

    # Sort the dictionary by keys - number of edges
    sorted_subgraphs = dict(sorted(subgraphs_by_edges.items(), reverse=True))

    # Convert the dictionary to a list of lists
    subgraphs_partitioned = list(sorted_subgraphs.values())

    return subgraphs_partitioned


def find_kmodule(G: nx.classes.graph.Graph, k: int, min_size: int, displays: bool = False) -> list:
    """
    Finds a set of k-modules in the given graph

    Arguments:
    G: The graph to check
    k: Size of each module
    min_size: Minimum number of k-modules the list should have
    displays: Display the progress

    Return:
    M: A list of k-modules or an empty list if none found
    """
    # Return an empty list if G has less vertices than 2k
    if G.number_of_nodes() < 2 * k: return []

    # Create a list of all possible subgraphs of size k in G -- |V(G)| comb k possibilities
    subgraphs = [list(subgraph) for subgraph in itertools.combinations(G.nodes, k)]
    num_subgraphs = len(subgraphs)

    # OPTION 1: Randomly shuffle the array to avoid divergence
    # random.shuffle(subgraphs)

    # OPTION 2: Sort the subraphs list by their number of edges descending
    # subgraphs = sorted(subgraphs, key=lambda x: nx.subgraph(G, x).number_of_edges(), reverse=True)

    # OPTION 3: Partition the subraphs list by their number of edges descending into list of lists
    subgraphs_partitioned = partition_subgraphs(G, subgraphs)


    total_checked = 0

    # Iterate over subgraph partitions
    for subgraphs in subgraphs_partitioned:

        # Look for each subgraph all possible k-modules
        for i in range(len(subgraphs)):
            sg1 = subgraphs[i]
            M = [sg1]

            for j in range(i + 1, len(subgraphs)):
                sg2 = subgraphs[j]
                if are_kmodule(G, M + [sg2]):
                    M.append(sg2)             

            # Displaying progress...
            if displays:
                # Increase the number of subgraphs checked and print
                total_checked += 1
                display_id = f'display_{k}'
                progress = total_checked / num_subgraphs
                message = f'Looking for k={k}: {total_checked}/{num_subgraphs} ({progress:.1%})'
                display(message, display_id=display_id, update=True)

            # If M has more subgraphs than the minimum number needed then the set is found
            if len(M) >= min_size:
                return M

    return []


def simplify_graph(G: nx.classes.graph.Graph, qr: int, displays: bool = False, k_max: int = 100) -> nx.classes.graph.Graph:
    """
    Algorithm for reducing the graph G according to the quantifier rank qr

    Arguments:
    G: The graph to check
    qr: Quantifier rank
    displays: Display the progress
    k_max: Maximum size of k-modules to look for - used for avoiding long runtimes

    Return:
    G2: Simplified graph
    """

    # Copy the graph
    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes)
    G2.add_edges_from(G.edges)

    # Start checking for k=1 modules
    k = 1

    while k <= G2.number_of_nodes() / (qr + 1) and k <= k_max:

         # Displaying progress...
        if displays:
            display_id = f'display_{k}'
            display("", display_id=display_id)

        has_kmodules = True
        while has_kmodules:

            # Checking for k-modules...
            M = find_kmodule(G2, k, min_size=qr + 1, displays=displays)

            if len(M) == 0:
                # When there is no modules then stop looking for k
                has_kmodules = False
            else:
                # Removing |M| - qr of those from G2
                to_remove = [node for subgraph in M[qr:] for node in subgraph]
                G2.remove_nodes_from(to_remove)

                if displays:
                    print(f'Removing {M[qr:]} from {M}')

        # Increase the k for the next step
        k += 1
        #print('-----------------------------------')

    # Print the number of removed vertices
    if displays:
        print('\n------------------------------------------------------------------\n')
        print(f'Removed {G.number_of_nodes() - G2.number_of_nodes()} vertices')

    return G2


def test1():
    display_id = 'display_1'
    display("", display_id=display_id)

    for i in range(100):
        message = f'{i}/100'
        display(message, display_id=display_id, update=True)
        print('newprint')








