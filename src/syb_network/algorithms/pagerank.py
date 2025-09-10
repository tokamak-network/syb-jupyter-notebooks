# pagerank_algorithm_plot.py

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys

# Import the PageRank-based scoring algorithm
from pagerank_scoring_algorithm import compute_next_scores, normalize_scores, compute_pagerank

def generate_random_graph(n, m):
    """
    Generates a random simple undirected graph with n vertices and m edges.
    """
    if n < 0 or m < 0:
        raise ValueError("Number of vertices and edges must be non-negative.")
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        raise ValueError(f"Edge count {m} exceeds the maximum possible of {max_edges}.")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    possible_edges = []
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                possible_edges.append((i, j))
    
    random.shuffle(possible_edges)
    edges_to_add = possible_edges[:m]
    G.add_edges_from(edges_to_add)
    
    return G

# ---

def display_graph_state(graph, scores, title):
    """
    Prints the graph's nodes, scores, and neighbors to the console.
    """
    print(f"\n--- {title} ---")
    if not graph.nodes():
        print("Graph is empty.")
        return
        
    # Create a mapping from node to index for consistent score lookup
    nodes = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for node in nodes:
        score = scores[node_to_idx[node]]
        neighbors = sorted(list(graph.neighbors(node)))
        print(f"Node {node:<2} | Score: {score:<7.4f} | Neighbors: {neighbors}")
    print(f"Total Score Sum: {np.sum(scores):.2f}")
    print("-" * (len(title) + 6))

# ---

def plot_graph_with_scores(graph, scores, title):
    """
    Plots the graph using a spring layout and displays node IDs and scores.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42) # For reproducible layout
    
    # Create labels with node ID and score
    # Ensure scores are correctly mapped to node IDs based on the graph's node list order
    nodes = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    labels = {node: f"{node}: {scores[node_to_idx[node]]:.2f}" for node in nodes}
    
    nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=800, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=11, font_color='black')
    
    plt.title(title, fontsize=16)
    plt.show()

# ---

def handle_user_edge_addition(graph, current_scores, sigma=1.0, alpha=0.15, max_pr_iterations=100):
    """
    Prompts the user to add an edge and displays the new normalized state using PageRank scoring.
    """
    n = graph.number_of_nodes()
    if n < 2:
        print("\nCannot add new edges to a graph with fewer than 2 vertices.")
        return graph, current_scores

    while True:
        try:
            prompt = f"\nEnter two distinct vertices to connect (from 0 to {n-1}), separated by a space: "
            user_input = input(prompt)
            i, j = map(int, user_input.split())

            if not (0 <= i < n and 0 <= j < n):
                print(f"Error: Vertices must be within the range [0, {n-1}].")
            elif i == j:
                print("Error: Vertices must be distinct.")
            elif graph.has_edge(i, j):
                print(f"Error: Edge ({i}, {j}) already exists.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter two integers separated by a space.")
    
    graph.add_edge(i, j)
    print(f"\nEdge ({i}, {j}) successfully added.")
    
    # Compute new scores using the PageRank-based algorithm
    new_scores = compute_next_scores(graph, current_scores, alpha=alpha, sigma=sigma, max_pr_iterations=max_pr_iterations)
    
    title = f"State After Adding Edge ({i}, {j}) (PageRank-based)"
    display_graph_state(graph, new_scores, title)
    plot_graph_with_scores(graph, new_scores, title)
    
    return graph, new_scores

def generate_graph_and_display(N_VERTICES, M_EDGES, sigma=1.0, alpha=0.15, max_pr_iterations=50):
    """
    Generates a random graph with specified vertices and edges, computes initial and updated scores,
    and displays the results. Returns the graph and updated scores.
    
    Args:
        N_VERTICES (int): Number of vertices in the graph
        M_EDGES (int): Number of edges in the graph
        sigma (float): Parameter for the PageRank scoring algorithm (default: 1.0)
        alpha (float): PageRank damping factor (default: 0.15)
        max_pr_iterations (int): Max iterations for PageRank convergence (default: 50)
    
    Returns:
        tuple: (graph, updated_scores)
    """
    # TODO: add a check for non positive N_VERTICES

    if N_VERTICES > 18:
        print(f"\n🚨 WARNING: You chose {N_VERTICES} vertices.")
        print("The scoring algorithm is very slow for N > 18 and may take a very long time.")
        print("Proceeding anyway...")

    max_possible_edges = N_VERTICES * (N_VERTICES - 1) // 2
    
    if not (0 <= M_EDGES <= max_possible_edges):
        raise ValueError(f"Number of edges must be between 0 and {max_possible_edges}.")

    G = generate_random_graph(N_VERTICES, M_EDGES)
    print(f"\n✅ Generated a random graph with {N_VERTICES} vertices and {M_EDGES} edges.")
    
    # Initial scores can be based on degree, or simply uniform
    degrees = np.array([G.degree(i) for i in sorted(G.nodes())], dtype=float)
    degree_sum = np.sum(degrees)

    if degree_sum > 0:
        initial_scores = degrees / degree_sum
    else:
        initial_scores = np.ones(N_VERTICES) / N_VERTICES

    initial_title = "Initial State (Scores from Normalized Degree)"
    display_graph_state(G, initial_scores, initial_title)
    plot_graph_with_scores(G, initial_scores, initial_title)

    print("\nComputing the first score update using PageRank-based algorithm...")
    updated_scores = compute_next_scores(G, initial_scores, alpha=alpha, sigma=sigma, max_pr_iterations=max_pr_iterations)
    updated_title = "Normalized State (First Iteration with PageRank-based Scoring)"
    display_graph_state(G, updated_scores, updated_title)
    plot_graph_with_scores(G, updated_scores, updated_title)
    
    return G, updated_scores
