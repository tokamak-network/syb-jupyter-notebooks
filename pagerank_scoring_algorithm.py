# pagerank_scoring_algorithm.py

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

def compute_pagerank(graph, alpha, source_node, max_iterations=100, tolerance=1e-6):
    """
    Computes a personalized PageRank vector for a given source node.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize pagerank vector, with all mass on the source node.
    s = np.zeros(n)
    if source_node in node_to_idx:
        s[node_to_idx[source_node]] = 1.0

    p_current = s.copy()
    
    # Pre-calculate degrees for efficiency
    degrees = dict(graph.degree())
    
    for iteration in range(max_iterations):
        p_next = np.zeros(n)
        
        # Calculate the random walk part
        for i in range(n):
            node_i = nodes[i]
            for neighbor in graph.neighbors(node_i):
                if degrees[node_i] > 0:
                    p_next[node_to_idx[neighbor]] += p_current[i] / degrees[node_i]

        # Apply the power iteration formula
        p_next = alpha * s + (1 - alpha) * 0.5 * (p_current + p_next)
        
        # Check for convergence
        if np.linalg.norm(p_next - p_current, 1) < tolerance:
            return normalize_scores(p_next)
            
        p_current = p_next.copy()
        
    return normalize_scores(p_current)

def compute_next_scores(graph, current_scores=None, alpha=0.15, sigma=1.0, max_pr_iterations=100):
    """
    Computes new scores based on Personalized PageRank.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Compute a score for each vertex
    for i, v in enumerate(nodes):
        # 1. Compute personalized pagerank vector for vertex v
        pr_vector = compute_pagerank(graph, alpha, v, max_pr_iterations)
        
        # 2. Get the permutation based on descending pagerank scores
        sorted_indices = np.argsort(pr_vector)[::-1]
        sorted_nodes = [nodes[idx] for idx in sorted_indices]
        
        min_ratio_for_v = float('inf')
        
        # 3. Iterate through subsets Sj
        for j in range(1, n + 1):
            Sj = set(sorted_nodes[:j])
            
            # Condition 1: i must be in Sj
            if v not in Sj:
                continue
                
            # Condition 2: Sum of scores in Sj < 1/2 of total score sum
            subset_score_sum = np.sum([x[node_to_idx[node]] for node in Sj])
            if subset_score_sum >= total_score_sum / 2:
                continue
                
            # If conditions are met, calculate the ratio
            num_boundary_edges = len(list(nx.edge_boundary(graph, Sj)))
            size_Sj = len(Sj)
            
            if size_Sj > 0:
                current_ratio = (num_boundary_edges ** sigma) / size_Sj
                min_ratio_for_v = min(min_ratio_for_v, current_ratio)
                
        y[i] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return normalize_scores(y)


def compute_next_scores_unnorm(graph, current_scores=None, alpha=0.15, sigma=1.0, max_pr_iterations=100):
    """
    Computes new scores based on Personalized PageRank without normalization.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Compute a score for each vertex
    for i, v in enumerate(nodes):
        # 1. Compute personalized pagerank vector for vertex v
        pr_vector = compute_pagerank(graph, alpha, v, max_pr_iterations)
        
        # 2. Get the permutation based on descending pagerank scores
        sorted_indices = np.argsort(pr_vector)[::-1]
        sorted_nodes = [nodes[idx] for idx in sorted_indices]
        
        min_ratio_for_v = float('inf')
        
        # 3. Iterate through subsets Sj
        for j in range(1, n + 1):
            Sj = set(sorted_nodes[:j])
            
            # Condition 1: i must be in Sj
            if v not in Sj:
                continue
                
            # Condition 2: Sum of scores in Sj < 1/2 of total score sum
            subset_score_sum = np.sum([x[node_to_idx[node]] for node in Sj])
            if subset_score_sum >= total_score_sum / 2:
                continue
                
            # If conditions are met, calculate the ratio
            num_boundary_edges = len(list(nx.edge_boundary(graph, Sj)))
            size_Sj = len(Sj)
            
            if size_Sj > 0:
                current_ratio = (num_boundary_edges ** sigma) / size_Sj
                min_ratio_for_v = min(min_ratio_for_v, current_ratio)
                
        y[i] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return y
