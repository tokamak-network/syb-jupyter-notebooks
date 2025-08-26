# initial_scores.py

import networkx as nx
import numpy as np
import random

def compute_initial_scores(graph, label):
    """
    Computes initial score vectors for graph nodes based on a specified label.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.array([])
    if label == "degree":
        scores = np.array([graph.degree(node) for node in sorted(graph.nodes())], dtype=float)
    elif label == "ones":
        scores = np.ones(n, dtype=float)
    elif label == "zeros":
        scores = np.zeros(n, dtype=float)
    elif label == "random":
        scores = np.random.rand(n)
    elif label == "increasing":
        scores = np.arange(n, dtype=float)
    else:
        raise ValueError(f"Unknown label '{label}'. Choose from 'degree', 'ones', 'zeros', 'random', 'increasing'.")
    return scores