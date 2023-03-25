# Model Checking with K-Modules
 A graph simplification algorithm that uses k-modules to improve the efficiency of model checking on graphs.

## Usage

1. The algorithm uses [NetworkX](https://networkx.org) library to model graphs. After installing the library simply create a graph or use one of the existing constructors in [graphs.py](src/graphs.py)

```python
import networkx as nx
from graphs import *

h = 3   # Height of the tree
chld = [2,3,4]   # Number of children at each height
G = create_tree(h=h, children=chld)
```

2. Call the algorithm to simplify the graph

```python
from algorithm import *
from IPython.display import display

qr = 2   # quantifier rank
displays = True   # Display the progress
G2 = simplify_graph(G, qr, displays)
```

3. Preferably you can plot the graphs by calling the following function from [plot.py](src/plot.py)

```python
from plot import *

title = 'An example of a simplified graph'
draw_graphs(G, G2, title)
```
