import networkx as nx
from pgmpy.estimators import PC
from graphviz import Digraph
import numpy as np
import time
import pandas as pd
import itertools
import sys
import os
from picause import confusion_matrix, oriented_confusion_matrix, pairlist2arrowstr, arrowstr2pairlist

def showBN(edges,file_name):
    node_attr = dict(
     style='filled',
     shape='box',
     align='left',
     fontsize='12',
     ranksep='0.1',
     height='0.2'
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    for a,b in edges:
        dot.edge(a,b)

    dot.render(filename=file_name, view = False)
    return dot   

def Adjacency_Performance(true_adj, est_adj):
    n = true_adj.shape[0]
    TP = 0
    FP = 0
    FN = 0
    combinations = list(itertools.combinations_with_replacement(list(range(n)), 2))
    true_adj_undirected = true_adj + true_adj.T
    est_adj_undirected = est_adj + est_adj.T

    true_adj_undirected = (true_adj_undirected > 0).astype(int)
    est_adj_undirected = (est_adj_undirected > 0).astype(int)
    for edge in combinations:
        u, v = edge
        # TP, the number of true adjacencies also present in the algorithm’s 
        # output graph;
        if true_adj_undirected[u][v]==1 and est_adj_undirected[u][v]==1:
            TP += 1
        
        # FP, the number of adjacencies in the algorithm’s 
        # output that do not correspond to true adjacencies; 
        if est_adj_undirected[u][v]==1 and true_adj_undirected[u][v]==0:
            FP += 1
        
        # FN, the number of true adjacencies that are not 
        # found in the algorithm’s output;
        if true_adj_undirected[u][v]==1 and est_adj_undirected[u][v]==0:
            FN += 1
    
    # sensitivity (TP/[TP + FN])
    sensitivity = TP / (TP + FN)
    # precision (TP/[TP + FP])
    precision = TP / (TP + FP)
    
    return sensitivity, precision

def Orientation_Performance(true_adj, est_adj):
    n = true_adj.shape[0]
    TP = 0
    FP = 0
    FN = 0
    combinations = list(itertools.product(list(range(n)), repeat = 2))
    for edge in combinations:
        u, v = edge
        # TP, the number of true edges that are present 
        # and oriented correctly in the output graph;
        if true_adj[u][v]==1 and est_adj[u][v]==1:
            TP += 1
        
        # FP, the number of oriented edges in the output graph that are 
        # absent or given the opposite directionality in the true graph; 
        if est_adj[u][v]==1 and true_adj[u][v]==0:
            FP += 1
        
        # FN, the number of true edges that are not present or 
        # not oriented correctly in the output graph;
        if true_adj[u][v]==1 and est_adj[u][v]==0:
            FN += 1
    
    # sensitivity (TP/[TP + FN])
    sensitivity = TP / (TP + FN)
    # precision (TP/[TP + FP])
    precision = TP / (TP + FP)
    
    return sensitivity, precision

# Funtion to evaluate the learned model structures.
def get_f1_score(estimated_model, path):
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_array(
        estimated_model, nodelist=nodes, weight=None
    )

    true_adj = np.load(path)

    recall_adjacency, precision_adjacency = Adjacency_Performance(true_adj, est_adj)

    recall_orientation, precision_orientation = Orientation_Performance(true_adj, est_adj)
    

    
    print("Recall of Adjacency: ", recall_adjacency)
    print("Precision of Adjacency: ", precision_adjacency)

    print("Recall of Orientation: ", recall_orientation)
    print("Precision of Orientation: ", precision_orientation)




if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    if len(sys.argv) != 5:
        print("Usage: python pc.py num_var num_edge num_sample n_jobs")
        sys.exit(1)

    num_var = int(sys.argv[1])
    num_edge = int(sys.argv[2])
    num_sample = int(sys.argv[3])
    n_jobs = int(sys.argv[4])

    dataset_dir = 'datasets/{}_{}_{}'.format(num_var, num_edge, num_sample)
    result_dir = dir = 'results/{}_{}_{}_{}'.format(num_var, num_edge, num_sample, n_jobs)
    os.makedirs(result_dir, exist_ok=True)

    # read data
    data_file = os.path.join(dataset_dir, 'data.csv')
    samples = pd.read_csv(data_file)

    # read edges
    edges_file = os.path.join(dataset_dir, 'edges.txt')
    true_edges = []
    with open(edges_file, 'r') as file:
        for line in file:
            node1, node2 = line.strip().split()
            true_edges.append((node1, node2))

    # estimate the graph with pc algorithm
    est = PC(data=samples)

    start_time = time.time()
    estimated_model = est.estimate(variant="parallel", max_cond_vars=2, n_jobs = n_jobs, ci_test= 'pearsonr', show_progress=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Estimation Cost Time:", elapsed_time, "seconds")

    estimated_edges = list(estimated_model.edges())

    est_txt_name = os.path.join(result_dir, 'est_edges.txt')
    with open(est_txt_name, 'w') as file:
        for edge in estimated_edges:
            file.write(f"{edge[0]} {edge[1]}\n")
    # evaluate
    cm = confusion_matrix(num_var, true_edges, estimated_edges)
    ocm = oriented_confusion_matrix(pairlist2arrowstr(true_edges), pairlist2arrowstr(estimated_edges))
    
    skeleton_precision = cm[0] / (cm[0] + cm[1])
    skeleton_sensitivity = cm[0] / (cm[0] + cm[3])

    oriented_precision = ocm['oriented_TP'] / (ocm['oriented_TP'] + ocm['oriented_FP'])
    oriented_sensitivity = ocm['oriented_TP'] / (ocm['oriented_TP'] + ocm['oriented_FN'])

    print("confusion_matrix: TP = {}, FP = {}, TN = {}, FN = {}".format(cm[0], cm[1], cm[2], cm[3]))
    print("oriented_confusion_matrix: TP = {}, FP = {}, FN = {}".format(ocm['oriented_TP'], ocm['oriented_FP'], ocm['oriented_FN']))

    print("skeleton: precision = {}, sensitivity = {}".format(skeleton_precision, skeleton_sensitivity))
    print("oriented: precision = {}, sensitivity = {}".format(oriented_precision, oriented_sensitivity))

    result_file = os.path.join(result_dir, 'experiment_data.txt')
    with open(result_file, 'w') as file:
        file.write(f"n_jobs: {n_jobs}\n")
        file.write(f"run_time: {elapsed_time}\n")
        file.write(f"skeleton_precision: {skeleton_precision}\n")
        file.write(f"skeleton_sensitivity: {skeleton_sensitivity}\n")
        file.write(f"oriented_precision: {oriented_precision}\n")
        file.write(f"oriented_sensitivity: {oriented_sensitivity}\n")
        file.write(f"confusion_matrix: {cm}\n")
        file.write("oriented_confusion_matrix:\n")
        for key, value in ocm.items():
            file.write(f"  {key}: {value}\n")

    

    estimated_graph = os.path.join(result_dir, 'estimated')
    true_graph = os.path.join(result_dir, 'true')
    showBN(estimated_edges, estimated_graph)
    showBN(true_edges, true_graph)

    print(f"Experiment result saved to {result_dir}")