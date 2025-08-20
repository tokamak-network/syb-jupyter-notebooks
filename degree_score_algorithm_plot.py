import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def compute_next_scores(graph, current_scores=None):
    """
    Computes and normalizes the next score for each vertex.
    """
    n = graph.number_of_nodes()
    
    if current_scores is None:
        # Initialize with 1/n if no scores are provided
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
        
    y = np.zeros(n)
    degrees = dict(graph.degree())
    
    for i in graph.nodes():
        for j in graph.neighbors(i):
            if degrees[j] > 0:
                y[i] += x[j] / degrees[j]

    # Normalize the new scores so they sum to 1
    score_sum = np.sum(y)
    if score_sum > 0:
        return y / score_sum
    
    return y # Return the zero vector if sum is 0

# ---

def display_graph_state(graph, scores, title):
    """
    Prints the graph's nodes, scores, and neighbors to the console.
    """
    print(f"\n--- {title} ---")
    if not graph.nodes():
        print("Graph is empty.")
        return
        
    for node in sorted(graph.nodes()):
        score = scores[node]
        neighbors = sorted(list(graph.neighbors(node)))
        print(f"Node {node:<2} | Score: {score:<7.4f} | Neighbors: {neighbors}")
    print(f"Total Score Sum: {np.sum(scores):.2f}") # Verify the sum is 1.0
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

def handle_user_edge_addition(graph, current_scores):
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
    
    new_scores = compute_next_scores(graph, current_scores)
    
    title = f"State After Adding Edge ({i}, {j})"
    display_graph_state(graph, new_scores, title)
    plot_graph_with_scores(graph, new_scores, title)
    
    return graph, new_scores

# ---

def generate_graph_and_display(N_VERTICES, M_EDGES):
    """Generates a random graph based on user input and displays its state.
    """
    
    if N_VERTICES < 0:
        print("Error: Number of vertices cannot be negative.")
        return
    #except ValueError:
    #    print("Invalid input. Please enter a whole number.")
            
    max_possible_edges = N_VERTICES * (N_VERTICES - 1) // 2
    
    
    if not (0 <= M_EDGES <= max_possible_edges):
        print(f"Error: Number of edges must be between 0 and {max_possible_edges}.")
        return
    #except ValueError:
    #    print("Invalid input. Please enter a whole number.")

    G = generate_random_graph(N_VERTICES, M_EDGES)
    print(f"\n✅ Generated a random graph with {N_VERTICES} vertices and {M_EDGES} edges.")
    
    # Initialize scores to 1/n for each node
    initial_scores = np.ones(N_VERTICES) / N_VERTICES
    initial_title = f"Initial State (Scores are 1/{N_VERTICES})"
    display_graph_state(G, initial_scores, initial_title)
    plot_graph_with_scores(G, initial_scores, initial_title)

    # Compute the first update (the result will be normalized)
    updated_scores = compute_next_scores(G, initial_scores)
    updated_title = "Normalized State (First Iteration)"
    display_graph_state(G, updated_scores, updated_title)
    plot_graph_with_scores(G, updated_scores, updated_title)
    
    print("\nGraph created. 👋")
    return G, updated_scores