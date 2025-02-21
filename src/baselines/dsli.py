import networkx as nx

def count_cycles_containing_edges(G):
    cycles = list(nx.simple_cycles(G))
    edge_cycle_count = {edge: 0 for edge in G.edges()}
    for edge in G.edges():
        for cycle in cycles:
            # Consider both directions for undirected comparison, if needed
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1])) + list(zip(cycle[-1:] + cycle[:-1], cycle))
            if edge in cycle_edges:
                edge_cycle_count[edge] += 1
        # Increment the count for each edge by 1 after checking all cycles
        edge_cycle_count[edge] += 1
    return edge_cycle_count

def calculate_node_strengths(G):
    out_strength, in_strength = {}, {}
    for node in G.nodes:
        out_strength[node] = sum(G[node][succ]['weight'] for succ in G.successors(node))
        in_strength[node] = sum(G[pred][node]['weight'] for pred in G.predecessors(node))
    return in_strength, out_strength

def bracket(node, in_strength, out_strength):
    return in_strength[node] + out_strength[node]

def calculate_importance(G, edge_cycle_count, in_strength, out_strength):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for x in list(G.successors(node)) + list(G.predecessors(node)):
            edge = (node, x) if (node, x) in G.edges else (x, node)
            cycle_count = edge_cycle_count[edge] if edge in edge_cycle_count else 0
            factor = bracket(node, in_strength, out_strength) + bracket(x, in_strength, out_strength) - 2 * G.get_edge_data(*edge)['weight']
            w = G.get_edge_data(*edge)['weight']
            omjer = bracket(node, in_strength, out_strength) / (bracket(node, in_strength, out_strength) + bracket(x, in_strength, out_strength))
            partial_umnozak = (cycle_count + 1) * factor * w * omjer
            product += partial_umnozak

        # Now, calculate the degrees sum and adjust the total umnozak
        strength_sum = in_strength[node] + out_strength[node]
        adjusted_umnozak = product + strength_sum
        
        importance_scores[node] = adjusted_umnozak  # Store the adjusted umnozak for normalization
        
        # Print the total umnozak and adjusted umnozak for each node
        # print(f"Total umnozak for Node {node}: {product}")
        # print(f"Total umnozak for Node {node} + in-degree + out-degree: {adjusted_umnozak}\n")

    # Normalize the scores
    suma = sum(importance_scores.values())
    normalized_scores = {node: score / suma * 100 for node, score in importance_scores.items()}
    return normalized_scores

def get_dsli_nodes(G):
    edge_cycle_count = count_cycles_containing_edges(G)
    in_strength, out_strength = calculate_node_strengths(G)
    importance_scores = calculate_importance(G, edge_cycle_count, in_strength, out_strength)
    dsli_dict = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    dsli_nodes = list(dsli_dict.keys())
    return dsli_nodes
