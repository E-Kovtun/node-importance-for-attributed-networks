from tqdm import tqdm 
import numpy as np
from src.simulation.LT_plus import run_LT
from src.simulation.IC_plus import run_IC
from src.simulation.SIR import run_SIR


def get_simulation_result(G, sorted_nodes, tw, aw, propagation_model, num_starts, num_runs):
    infl_mean = []
    infl_std = []
    for num in tqdm(num_starts):
        start_nodes = sorted_nodes[:num]
        all_infl = []
        if propagation_model == 'LT':
            for _ in range(num_runs):
                all_infl.append(len(run_LT(G, S=start_nodes, tw=tw, aw=aw)))
        elif propagation_model == 'IC':
            for _ in range(num_runs):
                all_infl.append(len(run_IC(G, S=start_nodes, tw=tw, aw=aw)))
        elif propagation_model == 'SIR':
            for _ in range(num_runs):
                all_infl.append(run_SIR(G, S=start_nodes))
        else:
            print('Unknowm propagation model')
        infl_mean.append(np.mean(np.array(all_infl)) / len(G))
        infl_std.append(np.std(np.array(all_infl) / len(G)))
    return infl_mean, infl_std
