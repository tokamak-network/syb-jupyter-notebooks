# argmax_scoring_algorithm.py

import networkx as nx
import numpy as np
import itertools
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds

def normalize_scores(scores):
    """
    Normalizes a vector of scores so they sum to 1.
    """
    score_sum = np.sum(scores)
    if score_sum > 0:
        return scores / score_sum
    return np.zeros_like(scores)

def compute_next_scores(graph, current_scores=None, sigma=2.0):
    """
    Computes new scores by first finding the maximum sum(y_i) over the polytope
    P_G,x with the y_V=x_V constraint, and then minimizing ||y-x||^2 over
    the solutions that achieve that maximum sum.

    P_G,x = {y in R>=0^V : y_S <= |partial S|^sigma for all S such that x_S < 1/2 x_V}
    where y_S = sum_{v in S} y_v, x_S = sum_{v in S} x_v, and x_V = sum_{v in V} x_v.

    WARNING: This function has exponential complexity (O(2^n)) due to generating
    and processing all subsets, and is only feasible for very small graphs (e.g., n < 20).
    Requires scipy for linear and quadratic programming.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_x_sum = np.sum(x)
    
    # If initial scores sum to zero, then all new scores must also be zero
    if total_x_sum == 0:
        # print("Debug: Initial scores sum to zero, returning zeros.")
        return np.zeros(n) # No need to normalize, it's already zeros

    # --- Step 1: Find the maximum achievable sum of y_i (max_sum_value) ---
    # Objective for LP: maximize sum(y_i) -> minimize -sum(y_i)
    c_lp = -np.ones(n)

    # Bounds for y_i: y_i >= 0
    # CORRECTED: linprog expects bounds as a sequence of (min, max) tuples
    bounds_lp_for_linprog = [(0, None)] * n 
    # Store bounds as a Bounds object for the minimize function (Step 2)
    bounds_for_minimize = Bounds(0, np.inf)

    # Inequality constraints: y_S <= |partial S|^sigma for subsets S where x_S < 1/2 x_V
    A_ub_lp = []
    b_ub_lp = []

    all_subsets_iterator = itertools.chain.from_iterable(
        itertools.combinations(nodes, r) for r in range(1, n + 1)
    )

    for subset_tuple in all_subsets_iterator:
        S = set(subset_tuple)
        x_S_sum = np.sum([x[node_to_idx[node]] for node in S])

        if x_S_sum < 0.5 * total_x_sum:
            row_A = np.zeros(n)
            for node in S:
                row_A[node_to_idx[node]] = 1.0
            A_ub_lp.append(row_A)
            
            num_boundary_edges = len(list(nx.edge_boundary(graph, S)))
            b_ub_lp.append(num_boundary_edges ** sigma)
    
    # Equality constraint for LP: sum(y_i) = sum(x_i) (i.e., y_V = x_V)
    A_eq_lp = [np.ones(n)]
    b_eq_lp = [total_x_sum]

    if A_ub_lp:
        A_ub_lp = np.array(A_ub_lp)
        b_ub_lp = np.array(b_ub_lp)
    else:
        A_ub_lp = np.empty((0, n))
        b_ub_lp = np.empty(0)

    A_eq_lp = np.array(A_eq_lp)
    b_eq_lp = np.array(b_eq_lp)

    try:
        # Use the corrected bounds format for linprog
        res_lp = linprog(c_lp, A_ub=A_ub_lp, b_ub=b_ub_lp, A_eq=A_eq_lp, b_eq=b_eq_lp, bounds=bounds_lp_for_linprog, method='highs')

        if not res_lp.success:
            print(f"DEBUG LP FAILED: {res_lp.message}")
            return normalize_scores(np.zeros(n))
        
        max_sum_value = -res_lp.fun
        y0_qp = res_lp.x 

    except Exception as e:
        print(f"DEBUG LP EXCEPTION: {e}")
        return normalize_scores(np.zeros(n))

    # --- Step 2: Minimize ||y-x||^2 subject to all constraints, including sum(y_i) = max_sum_value ---

    def objective_qp(y):
        return np.sum((y - x)**2)

    def jacobian_qp(y):
        return 2 * (y - x)

    constraints_qp = []

    if A_ub_lp.shape[0] > 0:
        constraints_qp.append(LinearConstraint(A_ub_lp, -np.inf, b_ub_lp))

    A_sum_y = np.ones((1, n))
    constraints_qp.append(LinearConstraint(A_sum_y, max_sum_value, max_sum_value))

    # Use the Bounds object for scipy.optimize.minimize
    bounds_qp = bounds_for_minimize 

    try:
        res_qp = minimize(objective_qp, y0_qp, method='SLSQP', jac=jacobian_qp,
                          bounds=bounds_qp, constraints=constraints_qp,
                          options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000})

        if res_qp.success:
            y_optimal = res_qp.x
            return normalize_scores(y_optimal)
        else:
            print(f"DEBUG QP FAILED: {res_qp.message}")
            if res_lp.success:
                return normalize_scores(res_lp.x)
            else:
                return normalize_scores(np.zeros(n))
    except Exception as e:
        print(f"DEBUG QP EXCEPTION: {e}")
        if res_lp.success:
            return normalize_scores(res_lp.x)
        else:
            return normalize_scores(np.zeros(n))


def compute_next_scores_unnorm(graph, current_scores=None, sigma=2.0):
    """
    Computes new scores by first finding the maximum sum(y_i) over the polytope
    P_G,x with the y_V=x_V constraint, and then minimizing ||y-x||^2 over
    the solutions that achieve that maximum sum, without final normalization.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    if current_scores is None:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_x_sum = np.sum(x)

    if total_x_sum == 0:
        # print("Debug: Initial scores sum to zero, returning zeros (unnorm).")
        return np.zeros(n)

    # --- Step 1: Find the maximum achievable sum of y_i (max_sum_value) ---
    c_lp = -np.ones(n)
    # CORRECTED bounds for linprog
    bounds_lp_for_linprog = [(0, None)] * n 
    bounds_for_minimize = Bounds(0, np.inf)

    A_ub_lp = []
    b_ub_lp = []

    all_subsets_iterator = itertools.chain.from_iterable(
        itertools.combinations(nodes, r) for r in range(1, n + 1)
    )

    for subset_tuple in all_subsets_iterator:
        S = set(subset_tuple)
        x_S_sum = np.sum([x[node_to_idx[node]] for node in S])

        if x_S_sum < 0.5 * total_x_sum:
            row_A = np.zeros(n)
            for node in S:
                row_A[node_to_idx[node]] = 1.0
            A_ub_lp.append(row_A)
            
            num_boundary_edges = len(list(nx.edge_boundary(graph, S)))
            b_ub_lp.append(num_boundary_edges ** sigma)
    
    A_eq_lp = [np.ones(n)]
    b_eq_lp = [total_x_sum]

    if A_ub_lp:
        A_ub_lp = np.array(A_ub_lp)
        b_ub_lp = np.array(b_ub_lp)
    else:
        A_ub_lp = np.empty((0, n))
        b_ub_lp = np.empty(0)

    A_eq_lp = np.array(A_eq_lp)
    b_eq_lp = np.array(b_eq_lp)

    try:
        # Use the corrected bounds format for linprog
        res_lp = linprog(c_lp, A_ub=A_ub_lp, b_ub=b_ub_lp, A_eq=A_eq_lp, b_eq=b_eq_lp, bounds=bounds_lp_for_linprog, method='highs')

        if not res_lp.success:
            print(f"DEBUG LP FAILED (unnorm): {res_lp.message}")
            return np.zeros(n)
        
        max_sum_value = -res_lp.fun
        y0_qp = res_lp.x

    except Exception as e:
        print(f"DEBUG LP EXCEPTION (unnorm): {e}")
        return np.zeros(n)

    # --- Step 2: Minimize ||y-x||^2 subject to all constraints, including sum(y_i) = max_sum_value ---
    def objective_qp(y):
        return np.sum((y - x)**2)

    def jacobian_qp(y):
        return 2 * (y - x)

    constraints_qp = []

    if A_ub_lp.shape[0] > 0:
        constraints_qp.append(LinearConstraint(A_ub_lp, -np.inf, b_ub_lp))

    A_sum_y = np.ones((1, n))
    constraints_qp.append(LinearConstraint(A_sum_y, max_sum_value, max_sum_value))

    bounds_qp = bounds_for_minimize

    try:
        res_qp = minimize(objective_qp, y0_qp, method='SLSQP', jac=jacobian_qp,
                          bounds=bounds_qp, constraints=constraints_qp,
                          options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000})

        if res_qp.success:
            return res_qp.x
        else:
            print(f"DEBUG QP FAILED (unnorm): {res_qp.message}")
            if res_lp.success:
                return res_lp.x
            else:
                return np.zeros(n)
    except Exception as e:
        print(f"DEBUG QP EXCEPTION (unnorm): {e}")
        if res_lp.success:
            return res_lp.x
        else:
            return np.zeros(n)
