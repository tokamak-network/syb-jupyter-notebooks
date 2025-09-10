import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- Dev Mode Functions ---

def generate_symmetric_vouch_table(num_nodes):
    """
    Generates a symmetric vouch table and a list of random balances.
    """
    if num_nodes < 0:
        raise ValueError("Number of nodes must be non-negative.")
    
    vouch_table = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.choice([True, False]):
                vouch_table[i][j] = 1
                vouch_table[j][i] = 1
                
    balance_list = [random.randint(1, 101) for _ in range(num_nodes)]
    
    return vouch_table, balance_list

def generate_asymmetric_vouch_table(num_nodes):
    """
    Generates an asymmetric vouch table and a list of random balances.
    """
    if num_nodes < 0:
        raise ValueError("Number of nodes must be non-negative.")
        
    vouch_table = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if random.choice([True, False]):
                vouch_table[i][j] = 1
                
    balance_list = [random.randint(1, 101) for _ in range(num_nodes)]
    
    return vouch_table, balance_list

# --- User Mode Functions ---

def generate_graph_from_vouch_table(vouch_table, balance_list, is_directed=False):
    """
    Creates and displays a graph from a vouch table and balance list.
    """
    num_nodes = len(vouch_table)
    if num_nodes == 0:
        print("Vouch table is empty. Cannot generate a graph.")
        return None

    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if vouch_table[i][j] == 1:
                G.add_edge(i, j)

    # Normalize balances to be used as scores
    total_balance = sum(balance_list)
    if total_balance > 0:
        initial_scores = np.array(balance_list) / total_balance
    else:
        initial_scores = np.zeros(num_nodes)

    title = "Initial Graph State from Vouch Table"
    display_graph_state(G, initial_scores, title)
    plot_graph_with_scores(G, initial_scores, title)
    
    return G

# --- Helper Functions (adapted from random_walk_scoring_algorithm.py) ---

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
    print(f"Total Score Sum: {np.sum(scores):.2f}")
    print("-" * (len(title) + 6))

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
