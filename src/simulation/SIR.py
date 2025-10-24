import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx


def run_SIR(G, S, beta=0.05, gamma=0.2, stable_steps=10):
    '''
    Input: G -- networkx directed graph
    S -- initial seed set of nodes
    '''

    model = ep.SIRModel(G)
    config = mc.Configuration()

    config.add_model_parameter('beta', beta) 
    config.add_model_parameter('gamma', gamma)  

    config.add_model_initial_configuration("Infected", S)

    model.set_initial_status(config)

    iteration_num = 0
    infected_counts = []

    while True:
    # while iteration_num <= 100:
        iteration = model.iteration() 
        iteration_num += 1

        status = iteration['status']

        infected_count = sum(1 for state in status.values() if state == 1)
        infected_counts.append(infected_count)

        if iteration_num >= stable_steps:
            recent_counts = infected_counts[-stable_steps:]
            if len(set(recent_counts)) == 1:
                break
    
        if (infected_count == 0):
            break

    influence_count = sum(1 for state in model.status.values() if (state == 1) or (state == 2))
    return influence_count 


# def run_SIR(G, S, tw, aw, alpha1=0.5, alpha2=0.5, beta=0.1, gamma=0.05, max_steps=100):
#     """
#     Runs an SIR diffusion process on graph G using NDlib,
#     with edge infection probabilities combined from topology and attribute weights.

#     Inputs:
#     - G: networkx.DiGraph
#     - S: list of initially infected nodes (seeds)
#     - tw: dict {(u,v): topology weight in [0,1]}
#     - aw: dict {(u,v): attribute weight in [0,1]}
#     - alpha1: weight for topology influence
#     - alpha2: weight for attribute influence
#     - beta: infection rate multiplier
#     - gamma: recovery rate
#     - max_steps: max simulation iterations

#     Output:
#     - infected_or_removed: set of nodes that were infected or removed by end of simulation
#     """

#     # Copy graph to avoid modifying original
#     G_copy = G.copy()

#     # Combine edge weights into 'transmission_prob' attribute and normalize per node
#     combined_weights = dict()
#     for v in G_copy.nodes():
#         in_edges = list(G_copy.in_edges(v))
#         total_weight = 0.0
#         for u, _ in in_edges:
#             p_topo = tw.get((u, v), 0)
#             p_attr = aw.get((u, v), 0)
#             p = alpha1 * p_topo + alpha2 * p_attr
#             combined_weights[(u, v)] = p
#             total_weight += p
#         if total_weight > 1.0:
#             for u, _ in in_edges:
#                 combined_weights[(u, v)] /= total_weight

#     for (u, v), p in combined_weights.items():
#         if G_copy.has_edge(u, v):
#             G_copy[u][v]['transmission_prob'] = p

#     # Instantiate NDlib SIR model
#     model = ep.SIRModel(G_copy)

#     # Configure model parameters
#     config = mc.Configuration()
#     config.add_model_parameter('beta', beta)
#     config.add_model_parameter('gamma', gamma)

#     # Set initial infected nodes (NDlib expects a dict {node: status})
#     initial_infected = {n: 1 for n in S}  # 1 = Infected status code
#     config.add_model_initial_configuration('Infected', initial_infected)

#     model.set_initial_status(config)

#     infected_or_removed = set(S)

#     for _ in range(max_steps):
#         iteration = model.iteration()
#         status = iteration[1]  # dict {node: status_code}

#         # NDlib status codes: Susceptible=0, Infected=1, Removed=2
#         for node, st in status.items():
#             if st == 1 or st == 2:
#                 infected_or_removed.add(node)

#         # Stop if no infected nodes remain
#         if all(st != 1 for st in status.values()):
#             break

#     return infected_or_removed


# def run_SIR(G, S, beta=0.05, gamma=0.2):
#     '''
#     Input: G -- networkx directed graph
#     S -- initial seed set of nodes
#     '''

#     model = ep.SIRModel(G)
#     config = mc.Configuration()

#     # Set model parameters: infection rate (beta) and recovery rate (gamma)
#     config.add_model_parameter('beta', beta)   # infection probability per contact
#     config.add_model_parameter('gamma', gamma)  # recovery probability

#     config.add_model_initial_configuration("Infected", S)

#     # Set the initial status in the model
#     model.set_initial_status(config)

#     # Run simulation until convergence (no new infections)
#     previous_infected_count = -1
#     iteration_num = 0

#     while True:
#         iteration = model.iteration()  # Execute a single iteration
#         iteration_num += 1

#         # Extract current status dictionary {node: status}
#         status = iteration['status']

#         # Count infected nodes (status code 1 = Infected)
#         infected_count = sum(1 for node, state in status.items() if state == 1)

#         # Check convergence: if no change in infected count, stop
#         if (infected_count == 0) or (infected_count == previous_infected_count):
#             break

#         previous_infected_count = infected_count

#     recovered_count = sum(1 for node, state in status.items() if state == 2)
#     return recovered_count 