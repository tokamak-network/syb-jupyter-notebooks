import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys

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

    # Normalize the final score vector so it sums to 1
    score_sum = np.sum(y)
    if score_sum > 0:
        return y / score_sum
    
    return y

# ---

def display_graph_state(graph, scores, title):
    """
    Prints the graph's nodes, scores, and neighbors to the console.
    """
    print(f"\n--- {title} ---")
    if not graph.nodes():
        print("Graph is empty.")
        return
        
    for node_idx, node in enumerate(sorted(graph.nodes())):
        score = scores[node_idx]
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
    pos = nx.spring_layout(graph, seed=42)
    
    labels = {i: f"{i}: {score:.2f}" for i, score in enumerate(scores)}
    
    nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=800, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=11, font_color='black')
    
    plt.title(title, fontsize=16)
    plt.show()

# ---

def handle_user_edge_addition(graph, current_scores, sigma=1.0):
    """
    Prompts the user to add an edge and displays the new normalized state.
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
    
    new_scores = compute_next_scores(graph, current_scores, sigma=sigma)
    
    title = f"State After Adding Edge ({i}, {j})"
    display_graph_state(graph, new_scores, title)
    plot_graph_with_scores(graph, new_scores, title)
    
    return graph, new_scores

def generate_graph_and_display(N_VERTICES, M_EDGES, sigma=1.0):
    """
    Generates a random graph with specified vertices and edges, computes initial and updated scores,
    and displays the results. Returns the graph and updated scores.
    
    Args:
        N_VERTICES (int): Number of vertices in the graph
        M_EDGES (int): Number of edges in the graph
        sigma (float): Parameter for the equal split scoring algorithm (default: 1.0)
    
    Returns:
        tuple: (graph, updated_scores)
    """
    #  TODO: add a check for non positive N_VERTICES

    if N_VERTICES > 18:
        print(f"\n🚨 WARNING: You chose {N_VERTICES} vertices.")
        print("The scoring algorithm is very slow for N > 18 and may take a very long time.")
        print("Proceeding anyway...")

    max_possible_edges = N_VERTICES * (N_VERTICES - 1) // 2
    
    if not (0 <= M_EDGES <= max_possible_edges):
        raise ValueError(f"Number of edges must be between 0 and {max_possible_edges}.")

    G = generate_random_graph(N_VERTICES, M_EDGES)
    print(f"\n✅ Generated a random graph with {N_VERTICES} vertices and {M_EDGES} edges.")
    
    degrees = np.array([G.degree(i) for i in sorted(G.nodes())], dtype=float)
    degree_sum = np.sum(degrees)

    if degree_sum > 0:
        initial_scores = degrees / degree_sum
    else:
        initial_scores = np.ones(N_VERTICES) / N_VERTICES

    initial_title = "Initial State (Scores from Normalized Degree)"
    display_graph_state(G, initial_scores, initial_title)
    plot_graph_with_scores(G, initial_scores, initial_title)

    print("\nComputing the first score update... (this may take a while for N > 12)")
    updated_scores = compute_next_scores(G, initial_scores, sigma=sigma)
    updated_title = "Normalized State (First Iteration)"
    display_graph_state(G, updated_scores, updated_title)
    plot_graph_with_scores(G, updated_scores, updated_title)
    
    return G, updated_scores