from algorithmNaive import simplify_graph as naive_simplify_graph
from algorithmRobust import simplify_graph as robust_simplify_graph
from graphs import *
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from IPython.display import display

def compare_on_random_graphs(sizes: list, p: float, qr: int, num_graphs_each: int, k_max: int = 5) -> (list, list):
    """
    Compares the naive and robust algorithms on random graphs

    Arguments:
    sizes: List of sizes of graphs to generate
    p: Probability of edge creation
    qr: Quantifier rank
    num_graphs_each: Number of graphs to generate for each size
    k_max: Maximum k to check for


    Return:
    time1, time2: Average times for each size for each algorithm
    """

    # Set a seed value in order to recreate graphs easily
    seed = 0

    # Generate random graphs
    graphs = []
    for n in sizes:
        graphs.append([create_random(n, p, seed) for _ in range(num_graphs_each)])
    total = len(sizes) * num_graphs_each

    # Run the naive algorithm
    time1 = []
    print("Running naive algorithm...")
    display_id1 = 'display_naive'
    num = 0
    for i in range(len(sizes)):
        time1.append([])
        for G in graphs[i]:
            # Display the progress
            message = f'Simplifying graph {num}/{total}'
            display(message, display_id=display_id1, update=True)
            num += 1
            # Run the algorithm
            start = time.time()
            naive_simplify_graph(G, qr, displays=True, k_max=k_max)
            end = time.time()
            time1[i].append(end - start)

        # Average the times
        time1[i] = np.mean(time1[i])
    print(f'Naive algorithm: {time1}')

    # Run the robust algorithm
    time2 = []
    print("Running robust algorithm...")
    display_id2 = 'display_robust'
    num = 0
    for i in range(len(sizes)):
        time2.append([])
        for G in graphs[i]:
            # Display the progress
            message = f'Simplifying graph {num}/{total}'
            display(message, display_id=display_id2, update=True)
            num += 1
            # Run the algorithm
            start = time.time()
            robust_simplify_graph(G, qr, displays=True, k_max=k_max)
            end = time.time()
            time2[i].append(end - start)

        # Average the times
        time2[i] = np.mean(time2[i])
    print(f'Robust algorithm: {time2}')

    return time1, time2


def compare_on_random_trees(sizes: list, qr: int, num_graphs_each: int, k_max: int = 5) -> (list, list):
    """
    Compares the naive and robust algorithms on random trees of different heights

    Arguments:
    sizes: List of sizes of trees to generate
    qr: Quantifier rank
    num_graphs_each: Number of graphs to generate for each height
    k_max: Maximum k to check for

    Return:
    time1, time2: Average times for each height for each algorithm
    """

    # Set a seed value in order to recreate graphs easily
    #seed = 0

    # Generate random trees
    graphs = []
    for s in sizes:
        graphs.append([nx.random_tree(s) for _ in range(num_graphs_each)])

    total = len(sizes) * num_graphs_each

    # Run the naive algorithm
    time1 = []
    print("Running naive algorithm...")
    display_id1 = 'display_naive'
    num = 0
    for i in range(len(sizes)):
        time1.append([])
        for G in graphs[i]:
            # Display the progress
            message = f'Simplifying graph {num}/{total}'
            display(message, display_id=display_id1, update=True)
            num += 1
            # Run the algorithm
            start = time.time()
            naive_simplify_graph(G, qr, displays=False, k_max=k_max)
            end = time.time()
            time1[i].append(end - start)

        # Average the times
        time1[i] = np.mean(time1[i])
    print(f'Naive algorithm: {time1}')

    # Run the robust algorithm
    time2 = []
    print("Running robust algorithm...")
    display_id2 = 'display_robust'
    num = 0
    for i in range(len(sizes)):
        time2.append([])
        for G in graphs[i]:
            # Display the progress
            message = f'Simplifying graph {num}/{total}'
            display(message, display_id=display_id2, update=True)
            num += 1
            # Run the algorithm
            start = time.time()
            robust_simplify_graph(G, qr, displays=False, k_max=k_max)
            end = time.time()
            time2[i].append(end - start)

        # Average the times
        time2[i] = np.mean(time2[i])
    print(f'Robust algorithm: {time2}')

    return time1, time2


def plot_time_comparison(time1: list, time2: list, sizes: list, title: str):
    """
    Plots the comparison of the naive and robust algorithms

    Arguments:
    time1, time2: List of times for each graph - on the y-axis
    sizes: List of sizes of graphs to generate - on the x-axis
    title: Title of the plot
    """

    # Plot the results
    plt.plot(sizes, time1, label="Naive algorithm")
    plt.plot(sizes, time2, label="Robust algorithm")
    plt.xlabel("Number of nodes")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.show()


# Main
if __name__ == "__main__":

    # Parameters
    sizes = list(range(10, 20))
    p = 0.7
    qr = 2
    num_graphs_each = 3
    k_max = 3

    # Run the comparison
    #time1, time2 = compare_on_random_graphs(sizes, p, qr, num_graphs_each, k_max=k_max)
    time1, time2 = compare_on_random_trees(sizes, qr, num_graphs_each, k_max=k_max)

    # Plot the results
    title = "Comparison of naive and robust algorithms on random trees"
    plot_time_comparison(time1, time2, sizes, title)