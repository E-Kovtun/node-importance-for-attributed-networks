from tqdm import tqdm 
import numpy as np
from src.simulation.LT_plus import run_LT
from src.simulation.IC_plus import run_IC


def get_simulation_result(G, sorted_nodes, tw, aw, propagation_model, num_starts, num_runs):
    infl = []
    for num in tqdm(num_starts):
        start_nodes = sorted_nodes[:num]
        all_infl = []
        if propagation_model == 'LT':
            for _ in range(num_runs):
                all_infl.append(len(run_LT(G, S=start_nodes, tw=tw, aw=aw)))
        elif propagation_model == 'IC':
            for _ in range(num_runs):
                all_infl.append(len(run_IC(G, S=start_nodes, tw=tw, aw=aw)))
        else:
            print('Unknowm propagation model')
        mean_infl = np.mean(all_infl) / len(G)
        infl.append(mean_infl)
    return infl
