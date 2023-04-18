import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G: nx.classes.graph.Graph, planar: bool = False) -> None:
    """
    Draws a graph

    Arguments:
    G: Graph to draw
    planar: Whether to draw in planar form

    Return:
    """

    # Create a new plot object
    fig = plt.figure(figsize=(12, 6))

    # Draw the graph
    if planar:
        nx.draw_planar(G, with_labels=True)
    else:
        nx.draw(G, with_labels=True)

    plt.show()


def draw_graphs(G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph, title: str = "", planar: bool = False) -> None:
    """
    Draws 2 graphs side to side

    Arguments:
    G1: First graph to draw
    G2: Second graph to draw
    qr: Quantifier rank to print
    title: Title of the plot
    planar: Whether to draw in planar form

    Return:
    """

    # Create a new plot object
    fig = plt.figure(figsize=(12, 6))

    # Draw the graphs
    subax1 = plt.subplot(121)
    if planar:
        nx.draw_planar(G1, with_labels=True)
    else:
        nx.draw(G1, with_labels=True)

    subax2 = plt.subplot(122)
    if planar:
        nx.draw_planar(G2, with_labels=True)
    else:
        nx.draw(G2, with_labels=True)

    # Name the plots
    if title:
        fig.suptitle(title)
    subax1.set_title('Original Graph')
    subax2.set_title('Simplified Graph')

    plt.show()
