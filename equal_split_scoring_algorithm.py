# equal_split_scoring_algorithm.py

import networkx as nx
import numpy as np
import itertools

def normalize_scores(scores):
    """
    Normalizes a vector of scores so they sum to 1.
    """
    score_sum = np.sum(scores)
    if score_sum > 0:
        return scores / score_sum
    return np.zeros_like(scores)

def compute_next_scores(graph, current_scores=None, sigma=1.0):
    """
    Computes new scores based on minimizing a ratio over subsets and normalizes them.
    The valid subsets are those that contain the node in question and whose sum of
    scoring values is at most half of the sum of all the scores in the graph.
    
    WARNING: This function has exponential complexity and is only feasible for
    very small graphs (e.g., n < 20).
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    
    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Iterate over each vertex to calculate its new score
    for v_idx, v in enumerate(nodes):
        min_ratio_for_v = float('inf')
        
        # Generate all non-empty subsets of vertices
        all_subsets = itertools.chain.from_iterable(
            itertools.combinations(nodes, r) for r in range(1, n + 1)
        )
        
        for subset_tuple in all_subsets:
            A = set(subset_tuple)
            
            # Condition 1: The subset A must contain the vertex v
            if v not in A:
                continue
            
            # Condition 2: The sum of scores in A must be at most half of the total score sum.
            subset_score_sum = np.sum([x[nodes.index(node)] for node in A])
            if subset_score_sum > total_score_sum / 2:
                continue
            
            # If conditions are met, calculate the ratio for this subset
            size_A = len(A)
            num_boundary_edges = len(list(nx.edge_boundary(graph, A)))
            
            current_ratio = (num_boundary_edges ** sigma) / size_A
            
            min_ratio_for_v = min(min_ratio_for_v, current_ratio)

        y[v_idx] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return normalize_scores(y)


def compute_next_scores_unnorm(graph, current_scores=None, sigma=1.0):
    """
    Computes new scores based on minimizing a ratio over subsets without normalization.
    The valid subsets are those that contain the node in question and whose sum of
    scoring values is at most half of the sum of all the scores in the graph.
    
    WARNING: This function has exponential complexity and is only feasible for
    very small graphs (e.g., n < 20).
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    
    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Iterate over each vertex to calculate its new score
    for v_idx, v in enumerate(nodes):
        min_ratio_for_v = float('inf')
        
        # Generate all non-empty subsets of vertices
        all_subsets = itertools.chain.from_iterable(
            itertools.combinations(nodes, r) for r in range(1, n + 1)
        )
        
        for subset_tuple in all_subsets:
            A = set(subset_tuple)
            
            # Condition 1: The subset A must contain the vertex v
            if v not in A:
                continue
            
            # Condition 2: The sum of scores in A must be at most half of the total score sum.
            subset_score_sum = np.sum([x[nodes.index(node)] for node in A])
            if subset_score_sum > total_score_sum / 2:
                continue
            
            # If conditions are met, calculate the ratio for this subset
            size_A = len(A)
            num_boundary_edges = len(list(nx.edge_boundary(graph, A)))
            
            current_ratio = (num_boundary_edges ** sigma) / size_A
            
            min_ratio_for_v = min(min_ratio_for_v, current_ratio)

        y[v_idx] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return y