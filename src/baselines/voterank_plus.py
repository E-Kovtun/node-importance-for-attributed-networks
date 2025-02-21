import networkx as nx
import math
from tqdm import tqdm
import numpy as np
from src.simulation.IC_plus import run_IC
from src.simulation.LT_plus import run_LT


def get_weight(G, degree_dic, rank):  # wij表示：i给j投票的能力，wij ！= wji, rank is seed noods
    weight = {}
    nodes = nx.nodes(G)
    rank_list = [i[0] for i in rank]
    for node in nodes:
        sum1 = 0
        neighbors = list(nx.neighbors(G, node))
        neighbors_common_rank = list(set(neighbors) & set(rank_list))
        if len(neighbors_common_rank) != 0:  # 节点与已选节点的直接为0
            for nc in neighbors_common_rank:
                weight[(node, nc)] = 0
        neighbours_without_rank = list(set(neighbors) - set(rank_list))  # voting for unselected nodes
        if len(neighbours_without_rank) != 0:  # if the node has other nieghbours
            for nbr in neighbours_without_rank:
                sum1 += degree_dic[nbr]
            for neigh in neighbours_without_rank:
                weight[(node, neigh)] = degree_dic[neigh] / sum1
        else:  # 当前节点只有已选节点作为邻居
            for neigh in neighbors:
                weight[(node, neigh)] = 0
    # for i, j in nx.edges(G):
    #     sum1 = 0
    #     for nbr in nx.neighbors(G, i):
    #         sum1 += degree_dic[nbr]
    #     weight[(i, j)] = degree_dic[j] / sum1
    return weight


def get_node_score2(G, nodesNeedcalcu, node_ability, degree_dic, rank):

    weight = get_weight(G, degree_dic, rank)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        neighbors = list(nx.neighbors(G, node))
        for nbr in neighbors:
            sum2 += node_ability[nbr] * weight[(nbr, node)]
        node_score[node] = math.sqrt(len(neighbors) * sum2)
    return node_score


def voterank_plus(G, l, lambdaa):
    '''

    :param G: use new indicator + lambda + voterank, the vote ability = log(dij)
    :param l: the number of spreaders
    :param lambdaa: retard infactor
    :return:
    '''
    rank = []

    # count dict
    nodes = list(nx.nodes(G))
    degree_li = nx.degree(G)
    d_max = max([i[1] for i in degree_li])
    degree_dic = {}
    for i in degree_li:
        degree_dic[i[0]] = i[1]

    # node's vote information
    node_ability = {}
    for item in degree_li:
        degree = item[1]
        node_ability[item[0]] = math.log(1 + (degree/d_max))  # ln(x)

    # node_ability_values = node_ability.values()
    # degree_values = degree_dic.values()
    # weaky = mean_value(node_ability_values) / mean_value(degree_values)
    # node's score
    node_score = get_node_score2(G, nodes, node_ability, degree_dic, rank)



    for i in range(l):
        # choose the max entropy node for the first time t aviod the error
        max_score_node, score = max(node_score.items(), key=lambda x: x[1])
        rank.append((max_score_node, score))
        # set the information quantity of selected nodes to 0
        node_ability[max_score_node] = 0
        # set entropy to 0

        node_score.pop(max_score_node)
        # for the max score node's neighbor conduct a neighbour ability surpassing
        cur_nbrs = list(nx.neighbors(G, rank[-1][0]))  # spreader's neighbour 1 th neighbors
        next_cur_neigh = []  # spreader's neighbour's neighbour 2 th neighbors
        for nbr in cur_nbrs:
            nnbr = nx.neighbors(G, nbr)
            next_cur_neigh.extend(nnbr)
            node_ability[nbr] *= lambdaa  # suppress the 1th neighbors' voting ability

        next_cur_neighs = list(set(next_cur_neigh))  # delete the spreaders and the 1th neighbors
        for ih in rank:
            if ih[0] in next_cur_neighs:
                next_cur_neighs.remove(ih[0])
        for i in cur_nbrs:
            if i in next_cur_neighs:
                next_cur_neighs.remove(i)

        for nnbr in next_cur_neighs:
            node_ability[nnbr] *= (lambdaa ** 0.5)  # suppress 2_th neighbors' voting ability
        # find the neighbor and neighbor's neighbor
        H = []
        H.extend(cur_nbrs)
        H.extend(next_cur_neighs)
        for nbr in next_cur_neighs:
            nbrs = nx.neighbors(G, nbr)
            H.extend(nbrs)

        H = list(set(H))
        for ih in rank:
            if ih[0] in H:
                H.remove(ih[0])
        new_nodeScore = get_node_score2(G, H, node_ability, degree_dic, rank)
        node_score.update(new_nodeScore)

        # choose the max entropy node
        # sorted_score_li = sorted(node_score.items(), key=operator.itemgetter(1), reverse=True)
        #
        #
        # late_max_shell = get_keys(k_shell_re, rank[-1][0])
        # shell_set = []
        #
        # for i in sorted_score_li:
        #     shell_set.append(get_keys(k_shell_re, i[0]))  # corresponding shell value in sorted list[node_shell, node_shell]
        # shell =next_f(late_max_shell, shell_set)



        # node = sorted_score_li[shell]
        # rank.append(node)
        # node_ability[node[0]] = 0
        # node_score.pop(node[0])


        # while len(sorted_score_li) != 0:
        #     if len(set(shell_set)) != 1:  # 存在不同shell的节点
        #         max_score_node = sorted_score_li[index][0]
        #         score = sorted_score_li[index][1]
        #         curr_max_shell = shell_set[index]
        #         if curr_max_shell != late_max_shell:  # 这轮选的点和上轮不在一个shell里
        #             rank.append((max_score_node, score))
        #             node_ability[max_score_node] = 0
        #             node_score.pop(max_score_node)
        #             break
        #         else:
        #             index += 1
        # else:
        #     max_score_node = sorted_score_li[index][0]
        #     score = sorted_score_li[index][1]
        #     rank.append((max_score_node, score))
        #     node_ability[max_score_node] = 0
        #     node_score.pop(max_score_node)


        # node_score.pop(max_score_node)

        # #set the information quantity of selected nodes to 0
        # node_ability[max_score_node] = 0
        # # set entropy to 0
        # node_score.pop(max_score_node)
        # print(i, rank)
    return rank


def get_voterank_simulation_result(G, tw, aw, propagation_model, num_starts, num_runs):
    G_undir = G.to_undirected()
    infl = []
    for l in tqdm(num_starts):
        voterank_res = voterank_plus(G_undir, l, 0.1)
        start_nodes = [t[0] for t in voterank_res]
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
