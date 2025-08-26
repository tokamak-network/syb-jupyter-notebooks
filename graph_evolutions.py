# graph_evolutions.py

import random
import networkx as nx
import itertools # Added for itertools.product

def DecreasingEvolution(graph, ITERATIONS):
    """
    Generates a sequence of graphs by removing one random edge at each step.
    
    Args:
        graph (nx.Graph): The initial graph.
        ITERATIONS (int): The number of graphs to generate after the initial one.
        
    Returns:
        list: A list of networkx.Graph objects representing the evolution.
    """
    evolution_list = [graph.copy()]
    current_graph = graph.copy()
    for _ in range(ITERATIONS):
        existing_edges = list(current_graph.edges())
        if not existing_edges:
            print("The graph is already totally disjoint (no more edges to remove). Stopping the evolution.")
            break
        edge_to_remove = random.choice(existing_edges)
        current_graph.remove_edge(*edge_to_remove)
        evolution_list.append(current_graph.copy())
    return evolution_list

def IncreasingEvolution(graph, ITERATIONS):
    """
    Generates a sequence of graphs by adding one random edge at each step.
    
    Args:
        graph (nx.Graph): The initial graph.
        ITERATIONS (int): The number of graphs to generate after the initial one.
        
    Returns:
        list: A list of networkx.Graph objects representing the evolution.
    """
    evolution_list = [graph.copy()]
    current_graph = graph.copy()
    n = current_graph.number_of_nodes()
    all_possible_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
    for _ in range(ITERATIONS):
        existing_edges = set(current_graph.edges())
        possible_new_edges = list(all_possible_edges - existing_edges)
        if not possible_new_edges:
            print("The graph is already complete. Stopping the evolution.")
            break
        new_edge = random.choice(possible_new_edges)
        current_graph.add_edge(*new_edge)
        evolution_list.append(current_graph.copy())
    return evolution_list

def RandomEvolution(N_VERTICES, ITERATIONS):
    """
    Generates a list of random graphs.
    Each graph has N_VERTICES vertices and a random number of edges.
    """
    evolution_list = []
    max_edges = N_VERTICES * (N_VERTICES - 1) // 2
    for _ in range(ITERATIONS):
        m_edges = random.randint(0, max_edges)
        G = nx.Graph()
        G.add_nodes_from(range(N_VERTICES))
        possible_edges = [(i, j) for i in range(N_VERTICES) for j in range(i + 1, N_VERTICES)]
        random.shuffle(possible_edges)
        edges_to_add = possible_edges[:m_edges]
        G.add_edges_from(edges_to_add)
        evolution_list.append(G)
    return evolution_list

def RandomEvolutionFromGraph(initial_graph, ITERATIONS):
    """
    Generates a sequence of random graphs with the same number of vertices as the initial graph.
    """
    n = initial_graph.number_of_nodes()
    evolution_list = [initial_graph.copy()]
    new_random_graphs = RandomEvolution(n, ITERATIONS)
    evolution_list.extend(new_random_graphs)
    return evolution_list

def StepRandomEvolution(graph, ITERATIONS):
    """
    Generates a sequence of graphs by randomly adding or removing a single edge at each step.
    """
    evolution_list = [graph.copy()]
    current_graph = graph.copy()
    n = current_graph.number_of_nodes()
    all_possible_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
    for _ in range(ITERATIONS):
        existing_edges = set(current_graph.edges())
        possible_new_edges = all_possible_edges - existing_edges
        can_add = bool(possible_new_edges)
        can_remove = bool(existing_edges)
        action = None
        if can_add and can_remove:
            action = random.choice(['add', 'remove'])
        elif can_add:
            action = 'add'
        elif can_remove:
            action = 'remove'
        else:
            print("The graph cannot be changed further. Stopping the evolution.")
            break
        if action == 'add':
            new_edge = random.choice(list(possible_new_edges))
            current_graph.add_edge(*new_edge)
        elif action == 'remove':
            edge_to_remove = random.choice(list(existing_edges))
            current_graph.remove_edge(*edge_to_remove)
        evolution_list.append(current_graph.copy())
    return evolution_list

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

def BridgeGraph(N, n, m, k, complete_subgraph_n=False, complete_subgraph_m=False):
    """
    Generates a random graph with two well-connected components (of n and m vertices)
    connected by a bridge of k edges, allowing for a total of N vertices.
    
    Args:
        N (int): The total number of vertices in the final graph.
        n (int): Number of vertices in the first component.
        m (int): Number of vertices in the second component.
        k (int): Number of bridge edges connecting the two components.
        complete_subgraph_n (bool): If True, makes the first subgraph a complete graph.
        complete_subgraph_m (bool): If True, makes the second subgraph a complete graph.

    Returns:
        nx.Graph: The generated bridge graph.
    """
    if N < n + m:
        raise ValueError(f"Total vertices N ({N}) must be greater than or equal to the sum of subgraph vertices (n+m = {n+m}).")
    if n < 0 or m < 0 or k < 0:
        raise ValueError("Number of vertices and edges must be non-negative.")
    if k > n * m:
        raise ValueError(f"Number of bridge edges ({k}) exceeds the maximum possible ({n*m}).")
    
    # Create the first component (n vertices)
    if complete_subgraph_n:
        G1 = nx.complete_graph(n)
    else:
        G1 = nx.Graph()
        G1.add_nodes_from(range(n))
        max_edges1 = n * (n - 1) // 2
        m1 = random.randint(max_edges1 // 2, max_edges1)  # Well-connected component
        possible_edges1 = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(possible_edges1)
        G1.add_edges_from(possible_edges1[:m1])

    # Create the second component (m vertices)
    if complete_subgraph_m:
        G2 = nx.complete_graph(m)
        G2 = nx.relabel_nodes(G2, {i: i + n for i in range(m)}) # Relabel to avoid overlap
    else:
        G2 = nx.Graph()
        G2.add_nodes_from(range(n, n + m))
        max_edges2 = m * (m - 1) // 2
        m2 = random.randint(max_edges2 // 2, max_edges2)
        possible_edges2 = [(i, j) for i in range(n, n + m) for j in range(i + 1, n + m)]
        random.shuffle(possible_edges2)
        G2.add_edges_from(possible_edges2[:m2])

    # Combine the two components and add isolated nodes if N > n+m
    G = nx.compose(G1, G2)
    isolated_nodes_start = n + m
    G.add_nodes_from(range(isolated_nodes_start, N))

    # Add k random bridge edges
    # Corrected: Using itertools.product instead of nx.cartesian_product
    possible_bridges = list(itertools.product(range(n), range(n, n + m)))
    random.shuffle(possible_bridges)
    
    bridge_edges = possible_bridges[:k]
    G.add_edges_from(bridge_edges)
    
    return G

def BridgeEvolution(
    initial_bridge_graph,
    n_nodes_in_comp_n,
    m_nodes_in_comp_m,
    iterations_n,
    iterations_m,
    iterations_star,
    iterations_bridge
):
    """
    Evolves a BridgeGraph by adding edges in specific regions.

    Args:
        initial_bridge_graph (nx.Graph): The starting BridgeGraph.
        n_nodes_in_comp_n (int): Number of nodes in the first component (used for identifying nodes).
        m_nodes_in_comp_m (int): Number of nodes in the second component (used for identifying nodes).
        iterations_n (int): Number of edges to attempt to add within the first component.
        iterations_m (int): Number of edges to attempt to add within the second component.
        iterations_star (int): Number of edges to attempt to add between components and extra nodes.
        iterations_bridge (int): Number of edges to attempt to add between the two main components.

    Returns:
        list: A list of networkx.Graph objects representing the evolution.
    """
    evolution_list = [initial_bridge_graph.copy()]
    current_graph = initial_bridge_graph.copy()
    N_total = current_graph.number_of_nodes()

    # Define node sets for clarity
    comp_n_nodes = set(range(n_nodes_in_comp_n))
    # Corrected: Changed n_nodes_in_comp_m to n_nodes_in_comp_n
    comp_m_nodes = set(range(n_nodes_in_comp_n, n_nodes_in_comp_n + m_nodes_in_comp_m))
    extra_nodes = set(range(n_nodes_in_comp_n + m_nodes_in_comp_m, N_total))

    # --- Step 1: Add edges within component n ---
    print(f"\n--- Adding {iterations_n} edges within component n ---")
    for _ in range(iterations_n):
        possible_new_edges_n = []
        for u in comp_n_nodes:
            for v in comp_n_nodes:
                if u < v and not current_graph.has_edge(u, v):
                    possible_new_edges_n.append((u, v))
        
        if not possible_new_edges_n:
            print("Component n is saturated. No more edges to add.")
            break
        
        new_edge = random.choice(possible_new_edges_n)
        current_graph.add_edge(*new_edge)
        evolution_list.append(current_graph.copy())

    # --- Step 2: Add edges within component m ---
    print(f"\n--- Adding {iterations_m} edges within component m ---")
    for _ in range(iterations_m):
        possible_new_edges_m = []
        for u in comp_m_nodes:
            for v in comp_m_nodes:
                if u < v and not current_graph.has_edge(u, v):
                    possible_new_edges_m.append((u, v))
        
        if not possible_new_edges_m:
            print("Component m is saturated. No more edges to add.")
            break
        
        new_edge = random.choice(possible_new_edges_m)
        current_graph.add_edge(*new_edge)
        evolution_list.append(current_graph.copy())

    # --- Step 3: Add edges between components (n or m) and extra nodes ---
    print(f"\n--- Adding {iterations_star} edges between components and extra nodes ---")
    for _ in range(iterations_star):
        possible_new_edges_star = []
        all_comp_nodes = comp_n_nodes.union(comp_m_nodes)
        
        for u in all_comp_nodes:
            for v in extra_nodes:
                if not current_graph.has_edge(u, v):
                    possible_new_edges_star.append((u, v))
        
        if not possible_new_edges_star:
            print("No more edges to add between components and extra nodes.")
            break
            
        new_edge = random.choice(possible_new_edges_star)
        current_graph.add_edge(*new_edge)
        evolution_list.append(current_graph.copy())

    # --- Step 4: Add edges between component n and component m ---
    print(f"\n--- Adding {iterations_bridge} edges between component n and component m ---")
    for _ in range(iterations_bridge):
        possible_new_edges_bridge = []
        for u in comp_n_nodes:
            for v in comp_m_nodes:
                if not current_graph.has_edge(u, v):
                    possible_new_edges_bridge.append((u, v))
        
        if not possible_new_edges_bridge:
            print("Bridge between component n and m is saturated. No more edges to add.")
            break
            
        new_edge = random.choice(possible_new_edges_bridge)
        current_graph.add_edge(*new_edge)
        evolution_list.append(current_graph.copy())

    return evolution_list
