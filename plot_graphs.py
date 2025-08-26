# plot_graphs.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

def plot_graph_evolution_with_scores(graphs, scores_list, title_prefix, layout_type="circular"):
    """
    Plots a sequence of graphs with fixed node positions and captions,
    displaying node labels and scores, with each graph inside a box.
    """
    if not graphs:
        print(f"No graphs to plot for {title_prefix}.")
        return

    num_graphs = len(graphs)
    cols = min(num_graphs, 5)
    rows = math.ceil(num_graphs / cols)
    
    num_nodes = graphs[0].number_of_nodes()
    dummy_graph = nx.complete_graph(num_nodes)
    
    if layout_type == "circular":
        pos = nx.circular_layout(dummy_graph)
    elif layout_type == "spring":
        pos = nx.spring_layout(dummy_graph, seed=42)
    elif layout_type == "random":
        pos = nx.random_layout(dummy_graph, seed=42)
    elif layout_type == "spectral": # Added spectral layout
        pos = nx.spectral_layout(dummy_graph)
    else:
        # Default to circular layout if an unknown type is provided
        pos = nx.circular_layout(dummy_graph)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_graphs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    is_random_evolution = "Random Evolution" in title_prefix

    for i, G in enumerate(graphs):
        ax = axes[i]
        
        # Create a dictionary for node labels showing ID and score
        labels = {node: f"{node}: {scores_list[i][node]:.2f}" for node in G.nodes()}
        
        # Get caption for the evolution
        caption = ""
        if is_random_evolution:
            caption = f"Iteration {i}"
        else:
            if i == 0:
                caption = "Initial Graph"
            else:
                prev_edges = set(graphs[i-1].edges())
                current_edges = set(G.edges())
                added_edges = list(current_edges - prev_edges)
                removed_edges = list(prev_edges - current_edges)
                
                if added_edges:
                    caption = f"Added edge {tuple(sorted(added_edges[0]))}"
                elif removed_edges:
                    caption = f"Removed edge {tuple(sorted(removed_edges[0]))}"
                else:
                    caption = "No change"
        
        # Draw the graph with fixed layout, node labels, and a single edge color
        nx.draw(G, pos, ax=ax, with_labels=False, node_color='lightgreen', node_size=500, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        
        ax.set_title(f"Step {i}")
        ax.text(0.5, -0.15, caption, ha='center', transform=ax.transAxes, fontsize=10)
        
        # Add the delimiting box
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
    for j in range(num_graphs, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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
    print("-" * (len(title) + 6))