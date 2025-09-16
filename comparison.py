# comparison.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

# --- Import the initial score generator ---
try:
    from initial_scores import compute_initial_scores
except ImportError:
    print("Error: Could not import 'compute_initial_scores' from 'initial_scores.py'.")
    print("Using a fallback function (normalized 'ones').")
    
    def compute_initial_scores(G, label="ones"):
        n = G.number_of_nodes()
        if label == "ones":
            return np.ones(n)
        elif label == "degree":
            nodes = sorted(G.nodes())
            degrees = np.array([G.degree(i) for i in nodes], dtype=float)
            return degrees
        return np.ones(n)

# --- Import the four scoring functions ---
from random_walk_scoring_algorithm import (
    compute_next_scores as compute_random_walk,
)
from pagerank_scoring_algorithm import (
    compute_next_scores as compute_pagerank
)
from equal_split_scoring_algorithm import (
    compute_next_scores as compute_equal_split
)
from argmax_scoring_algorithm import (
    compute_next_scores as compute_argmax
)

# --- Helper Function to Generate Random Graphs ---

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

# --- Helper Function for Plotting a Single Graph ---

def plot_graph_with_scores(G, pos, scores, title, ax=None):
    """
    Plots a graph on a given matplotlib axes 'ax' or creates a new figure.
    Uses a fixed layout 'pos' to ensure nodes are in the same place.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    nodes = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    labels = {}
    for node in nodes:
        # Added a check for safety
        if node_to_idx.get(node, -1) >= len(scores):
             labels[node] = f"{node}: N/A"
             continue
        score_val = scores[node_to_idx[node]]
        if np.isnan(score_val):
            labels[node] = f"{node}: N/A"
        else:
            labels[node] = f"{node}: {score_val:.3f}"
    
    nx.draw(G, pos, ax=ax, with_labels=False, node_color='skyblue', node_size=700, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10, font_color='black')
    ax.set_title(title, fontsize=14)
    
    if show_plot:
        plt.show()

# --- Helper Function for Text Output ---

def print_score_comparison(nodes, initial, rw, pr, es, am):
    """Prints a formatted table of all scores to the console."""
    print("\n\n" + "="*70)
    print(" " * 25 + "SCORE COMPARISON")
    print("="*70)
    
    header = f"{'Node':<6} | {'Input':<10} | {'RandomWalk':<10} | {'PageRank':<10} | {'EqualSplit':<10} | {'Argmax':<10}"
    print(header)
    print("-" * len(header))
    
    for i, node in enumerate(nodes):
        s_init = f"{initial[i]:.4f}"
        s_rw = f"{rw[i]:.4f}"
        s_pr = f"{pr[i]:.4f}"
        
        s_es = f"{es[i]:.4f}" if not np.isnan(es[i]) else "N/A"
        s_am = f"{am[i]:.4f}" if not np.isnan(am[i]) else "N/A"
        
        print(f"{node:<6} | {s_init:<10} | {s_rw:<10} | {s_pr:<10} | {s_es:<10} | {s_am:<10}")
        
    print("-" * len(header))
    sum_init = np.sum(initial)
    sum_rw = np.sum(rw)
    sum_pr = np.sum(pr)
    sum_es = np.nansum(es)
    sum_am = np.nansum(am)

    s_sum_es = f"{sum_es:<10.2f}" if sum_es > 0 else "N/A       "
    s_sum_am = f"{sum_am:<10.2f}" if sum_am > 0 else "N/A       "

    print(f"{'SUM':<6} | {sum_init:<10.2f} | {sum_rw:<10.2f} | {sum_pr:<10.2f} | {s_sum_es} | {s_sum_am}")
    print("=" * 70 + "\n")

# --- Helper Function for Bar Plot ---

def plot_score_barplot(nodes, initial, rw, pr, es, am):
    """Plots a grouped bar chart comparing all scores for each node."""
    print("Generating score comparison bar plot...")
    
    n_nodes = len(nodes)
    
    algos_data = {
        'Input': initial,
        'RandomWalk': rw,
        'PageRank': pr
    }
    if not np.isnan(es[0]):
        algos_data['EqualSplit'] = es
    if not np.isnan(am[0]):
        algos_data['Argmax'] = am
        
    algo_names = list(algos_data.keys())
    n_algos_to_plot = len(algo_names)

    x = np.arange(n_nodes)
    width = 0.8 / n_algos_to_plot
    fig, ax = plt.subplots(figsize=(max(12, n_nodes * 1.5), 7))
    
    offsets = np.linspace(-width * (n_algos_to_plot - 1) / 2, 
                           width * (n_algos_to_plot - 1) / 2, 
                           n_algos_to_plot)

    for i, (name, data) in enumerate(algos_data.items()):
        offset = offsets[i]
        rects = ax.bar(x + offset, data, width, label=name)
        ax.bar_label(rects, padding=3, fmt='%.3f', rotation=90, fontsize=8)

    ax.set_ylabel('Scores')
    ax.set_title('Score Comparison by Node and Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.set_xlabel('Node ID')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    all_scores = np.concatenate(list(algos_data.values()))
    ax.set_ylim(bottom=0, top=np.nanmax(all_scores) * 1.25)

    fig.tight_layout()
    plt.show()


# --- Comparison Function ---

def compare(n_vertices,
            m_edges,
            sigma_equal_split=2.0, 
            sigma_argmax=2.0, 
            sigma_pagerank=2.0,
            alpha_pagerank=0.15,  
            max_pr_iterations=100,
            existing_graph=None,
            previous_scores_dict=None): 
    
    # --- Parameters ---
    N_VERTICES = n_vertices
    M_EDGES =  m_edges
    SIGMA_EQUAL_SPLIT = sigma_equal_split
    SIGMA_PAGERANK = sigma_pagerank
    ALPHA_PAGERANK = alpha_pagerank
    MAX_PR_ITERATIONS = max_pr_iterations
    SIGMA_ARGMAX = sigma_argmax
    # --------------------

    # --- 1. Get Graph and Input Scores ---
    
    if existing_graph is None:
        # First run: Generate graph and initial scores
        print(f"Generating a random graph with {N_VERTICES} vertices and {M_EDGES} edges...")
        max_edges = N_VERTICES * (N_VERTICES - 1) // 2
        if M_EDGES > max_edges:
            print(f"Warning: M_EDGES ({M_EDGES}) exceeds max possible ({max_edges}). Setting to max.")
            M_EDGES = max_edges
        if N_VERTICES <= 0:
            print("Error: N_VERTICES must be positive.")
            return None, None
        
        G = generate_random_graph(N_VERTICES, M_EDGES)
        nodes = sorted(G.nodes())
        
        # This is the "fresh start" score
        initial_scores = compute_initial_scores(G, label="ones")
        score_sum = np.sum(initial_scores)
        if score_sum > 0:
            initial_scores = initial_scores / score_sum # Normalize
        else:
            initial_scores = np.ones(N_VERTICES) / N_VERTICES if N_VERTICES > 0 else np.array([])
        
        # On first run, all algos start from the same initial scores
        input_rw = initial_scores
        input_pr = initial_scores
        input_es = initial_scores
        input_am = initial_scores
        display_initial_scores = initial_scores
        print("Generated initial scores (normalized 'ones').")

    else:
        # Subsequent run: Use existing graph and previous scores
        print("Using existing graph and previous scores as input.")
        G = existing_graph
        nodes = sorted(G.nodes())
        N_VERTICES = G.number_of_nodes() # Update N_VERTICES
        
        fallback_scores = np.ones(N_VERTICES) / N_VERTICES if N_VERTICES > 0 else np.array([])
        
        # Each algorithm starts from its *own* previous output
        input_rw = previous_scores_dict.get('rw', fallback_scores)
        input_pr = previous_scores_dict.get('pr', fallback_scores)
        input_es = previous_scores_dict.get('es', fallback_scores)
        input_am = previous_scores_dict.get('am', fallback_scores)
        
        # Handle cases where a previous algo was skipped (resulted in NaN)
        if np.isnan(input_rw).any(): input_rw = fallback_scores
        if np.isnan(input_pr).any(): input_pr = fallback_scores
        if np.isnan(input_es).any(): input_es = fallback_scores
        if np.isnan(input_am).any(): input_am = fallback_scores
        
        # "Initial" column will show the "ones" vector for this new graph state
        display_initial_scores = compute_initial_scores(G, label="ones")
        score_sum = np.sum(display_initial_scores)
        if score_sum > 0:
            display_initial_scores = display_initial_scores / score_sum
        else:
            display_initial_scores = fallback_scores

    # --- 2. Run All Scoring Algorithms (NO SAFETY CHECKS) ---
    
    print("\n--- Running Scoring Algorithms ---")

    # 1. Random Walk
    print("1. Computing Random Walk scores...")
    scores_rw = compute_random_walk(G, input_rw)

    # 2. PageRank-based
    print("2. Computing PageRank-based scores...")
    scores_pr = compute_pagerank(G, input_pr,
                                 alpha=ALPHA_PAGERANK,
                                 sigma=SIGMA_PAGERANK,
                                 max_pr_iterations=MAX_PR_ITERATIONS)
    
    # 3. Equal Split (Slow, O(2^n))
    print(f"3. Computing Equal Split scores (N={N_VERTICES})... this may take a while.")
    scores_es = compute_equal_split(G, input_es, sigma=SIGMA_EQUAL_SPLIT)

    # 4. Argmax (Very Slow, O(2^n))
    print(f"4. Computing Argmax scores (N={N_VERTICES})... this will be slow.")
    scores_am = compute_argmax(G, input_am, sigma=SIGMA_ARGMAX)
        
    print("--- All computations complete. ---")

    # --- 3. Print Text Comparison Table ---
    
    print_score_comparison(nodes, display_initial_scores, scores_rw, scores_pr, scores_es, scores_am)

    # --- 4. Generate Plots ---
    
    print("Generating plots...")
    
    pos = nx.spring_layout(G, seed=42) 
    
    plot_graph_with_scores(G, pos, display_initial_scores, "Input Scores ('Ones')")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Algorithm Score Comparison", fontsize=20)
    
    plot_graph_with_scores(G, pos, scores_rw, "Random Walk", ax=axes[0])
    plot_graph_with_scores(G, pos, scores_pr, "PageRank-based", ax=axes[1])
    
    # Handle skipped plots visually (will only trigger if compute fails and returns NaN)
    if np.isnan(scores_es).any():
        axes[2].set_title("Equal Split (SKIPPED/FAILED)", fontsize=14)
        axes[2].text(0.5, 0.5, f"SKIPPED or FAILED", ha='center', va='center', fontsize=12, color='red')
        axes[2].axis('off')
    else:
        plot_graph_with_scores(G, pos, scores_es, "Equal Split", ax=axes[2])

    if np.isnan(scores_am).any():
        axes[3].set_title("Argmax (SKIPPED/FAILED)", fontsize=14)
        axes[3].text(0.5, 0.5, f"SKIPPED or FAILED", ha='center', va='center', fontsize=12, color='red')
        axes[3].axis('off')
    else:
        plot_graph_with_scores(G, pos, scores_am, "Argmax", ax=axes[3])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plot_score_barplot(nodes, display_initial_scores, scores_rw, scores_pr, scores_es, scores_am)

    print("\n--- Comparison script finished. ---")
    
    new_scores_dict = {
        'rw': scores_rw,
        'pr': scores_pr,
        'es': scores_es,
        'am': scores_am
    }
    
    return G, new_scores_dict

# --- Function to Handle User Input & Loop ---

def handle_user_edge_addition(graph, 
                              scores_dict, 
                              n_vertices, 
                              m_edges,
                              sigma_equal_split=2.0,
                              sigma_argmax=2.0, 
                              sigma_pagerank=2.0,
                              alpha_pagerank=0.15,  
                              max_pr_iterations=100):
    """
    Handles the main interactive loop.
    Prompts the user to add an edge, then re-runs the comparison.
    """

    params = {
        'sigma_equal_split': sigma_equal_split,
        'sigma_argmax': sigma_argmax,
        'sigma_pagerank': sigma_pagerank,
        'alpha_pagerank': alpha_pagerank,
        'max_pr_iterations': max_pr_iterations}
    
    current_graph = graph
    current_scores_dict = scores_dict

    while True:
        n = current_graph.number_of_nodes()
        if n < 2:
            print("\nGraph has fewer than 2 nodes, cannot add edges.")
            return

        # --- Inner loop for user input validation ---
        while True:
            prompt = f"\nEnter two distinct vertices to connect (from 0 to {n-1}), or 'q' to quit: "
            user_input = input(prompt)
            
            if user_input.lower() == 'q':
                return # Exit the main loop
                
            try:
                i, j = map(int, user_input.split())

                if not (0 <= i < n and 0 <= j < n):
                    print(f"Error: Vertices must be within the range [0, {n-1}].")
                elif i == j:
                    print("Error: Vertices must be distinct.")
                elif current_graph.has_edge(i, j):
                    print(f"Error: Edge ({i}, {j}) already exists.")
                else:
                    # Valid edge, break inner loop
                    break 
            except ValueError:
                print("Invalid input. Please enter two integers separated by a space.")
            except Exception as e:
                print(f"An error occurred: {e}")

        # --- Add edge and re-run comparison ---
        current_graph.add_edge(i, j)
        print(f"\n✅ Edge ({i}, {j}) successfully added.")
        print(f"\n--- Re-running comparison on updated graph ---")
        
        # Call compare again, passing the modified graph AND previous scores
        current_graph, current_scores_dict = compare(
            n_vertices, m_edges, **params, 
            existing_graph=current_graph,
            previous_scores_dict=current_scores_dict
        )

# --- Main Application Loop ---

def main():
    """
    Main function to run the graph comparison application.
    """
    # --- Parameters ---
    N_VERTICES = 10
    M_EDGES = 12
    
    params = {
        'sigma_equal_split': 2.0,
        'sigma_argmax': 2.0,
        'sigma_pagerank': 2.0,
        'alpha_pagerank': 0.15,
        'max_pr_iterations': 100
    }

    try:
        # --- 1. Run Initial Comparison (Generates Graph) ---
        current_graph, current_scores_dict = compare(
            N_VERTICES, M_EDGES, **params, 
            existing_graph=None, 
            previous_scores_dict=None
        )
        if current_graph is None:
            print("Failed to initialize graph.")
            return

        # --- 2. Start Interactive Loop ---
        handle_user_edge_addition(
            current_graph, 
            current_scores_dict,
            N_VERTICES, 
            M_EDGES,
            **params
            )

        print("\nExiting application. Goodbye! 👋")

    except KeyboardInterrupt:
        print("\nCaught interrupt, exiting. Goodbye! 👋")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")