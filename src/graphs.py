import networkx as nx
import random
import itertools

def create_complete(n) -> nx.classes.graph.Graph:
    """
    Creates a complete graph of size n

    Arguments:
    n: Size of the graph

    Returns:
    G: Resulting graph
    """

    # Create the graph
    G = nx.Graph()

    # Nodes and edges
    nodes_list = list(range(n))
    edge_list = list(itertools.combinations(nodes_list, 2))

    G.add_nodes_from(nodes_list)
    G.add_edges_from(edge_list)

    return G


def create_tree(h: int, children: list = [], rmv: list = []) -> nx.classes.graph.Graph:
    """
  Creates a tree in a simple way

  Arguments:
  h: Height of the tree
  children: List of number of children for a parent at each height
  rmv: A list of nodes to remove from the tree

  Returns:
  T: Resulting tree
  """

    # If children isn't specified then create a random children (0-3) list
    if len(children) == 0:
        children = [random.randint(1, 3) for i in range(h)]

    # Check if the number of children corresponds to height
    if len(children) != h:
        raise ValueError("Length of children list doesn't match the height, should be h!")

    # Create the tree
    T = nx.Graph()

    # Create a list of id's of nodes at each height
    nodes_list = [[0]] + [[None]] * (h)
    last_id = 1
    for i in range(1, h + 1):
        # Number of nodes and parents for this height
        num_nodes = children[i - 1]
        num_parent = len(nodes_list[i - 1])

        # Add the lists of id's
        start_id = last_id
        last_id = start_id + (num_nodes * num_parent)
        nodes_list[i] = list(range(start_id, last_id))

    # Add the nodes
    T.add_nodes_from([item for sublist in nodes_list for item in sublist])

    # Add the edges
    for p_id, nodes in enumerate(nodes_list[1:]):
        # Number of parents and children for each parent
        num_parents = len(nodes_list[p_id])
        num_child = int(len(nodes) / num_parents)

        # Children for each parent
        childs_for_parent = [nodes[i:i + num_child] for i in range(0, len(nodes), num_child)]

        # Edges for this height to the parents
        edges = [(parent, node) for id, parent in enumerate(nodes_list[p_id]) for node in childs_for_parent[id]]
        T.add_edges_from(edges)

    # Check if the nodes in rmv exist
    if any([not node in T.nodes for node in rmv]):
        raise ValueError("Node doesn't exist!")

    # Remove the nodes
    T.remove_nodes_from(rmv)

    # Remove the unconnected subgraphs
    if not nx.is_connected(T):
        # Get the largest connected component
        largest_comp = max(nx.connected_components(T), key=len)
        T = nx.induced_subgraph(T, largest_comp)

    return T


def create_random_tree(h: int, seed: int = 0) -> nx.classes.graph.Graph:
    """
    Creates a random tree in a simple way
    Args:
        h: Height of the tree
        seed: Seed value

    Returns:
        T: Resulting tree
    """

    # Set the seed
    random.seed(seed)

    # Create a list of number of children for a parent at each height
    children = [random.randint(1, 3) for i in range(h)]

    # Create a list of nodes to remove by picking some random nodes
    num_rmv = random.randint(0, int(h/2))
    rmv = random.sample(range(1, 2**h), num_rmv)

    # Create the tree
    T = create_tree(h, children, rmv)

    return T


def create_kpartite(sizes: tuple, complete: bool = False) -> nx.classes.graph.Graph:
    """
    Creates a k-partite graph in a simple way

    Arguments:
    sizes: A tuple with the sizes of each partition set

    Returns:
    G: Resulting graph
    """

    # Create the tree
    G = nx.Graph()

    #  Create the nodes list in a list and add to the Graph
    nodes = []
    lastId = 0
    for size in sizes:
        partit = list(range(lastId, lastId + size))
        nodes.append(partit)

        # Add the nodes
        G.add_nodes_from(partit)

        # Increment the lastId for the next iteratiton
        lastId = lastId + size

    # Create and add the edges list
    # Generate all possible pairs of lists
    list_pairs = list(itertools.combinations(nodes, 2))

    # Generate all possible tuples of unique elements from the paired lists
    tuples = []
    for pair in list_pairs:
        for elem1 in pair[0]:
            for elem2 in pair[1]:
                if elem1 != elem2:
                    tuples.append(tuple(sorted([elem1, elem2])))

    # Get unique tuples
    unique_tuples = list(set(tuples))

    # Shuffle the list of unique tuples
    random.shuffle(unique_tuples)

    # Cut some of them
    if not (complete):
        cutId = random.randint(0, int(len(unique_tuples) / 2))
        unique_tuples = unique_tuples[cutId:]

    # Add the edges
    G.add_edges_from(unique_tuples)

    return G


def create_random(n: int, p: float, seed: int = 0) -> nx.classes.graph.Graph:
    """
    Creates a random graph in a simple way using a seed value

    Arguments:
    n: Number of nodes
    p: Probability of an edge
    seed: Seed value

    Returns:
    G: Resulting graph
    """

    # Create the graph
    G = nx.Graph()

    # Add the nodes
    G.add_nodes_from(list(range(n)))

    # Set the seed
    random.seed(seed)

    # Add the edges
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)

    return G


