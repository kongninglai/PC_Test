from picause import StructuralEquationDagModel
import numpy as np
import os
import sys

def generate_graph(num_var, num_edges):
    sem = StructuralEquationDagModel(num_var=num_var, num_edges=num_edges)
    overflow = 0
    while(sem.test_residual_overflow() and overflow < 100):
        overflow += 1
        sem = StructuralEquationDagModel(num_var=num_var, num_edges=num_edges)
    if (overflow == 100):
        print("Failed to find a valid graph!")
        sys.exit(1)
    return sem

if __name__ == "__main__":
        
    if len(sys.argv) != 4:
        print("Usage: python generate_data.py num_var num_edge num_sample")
        sys.exit(1)

    # read arguments
    num_var = int(sys.argv[1])
    num_edge = int(sys.argv[2])
    num_sample = int(sys.argv[3])

    # generate a valid graph
    sem = generate_graph(num_var, num_edge)

    # create a directory
    dir = 'datasets/{}_{}_{}'.format(num_var, num_edge, num_sample)
    os.makedirs(dir, exist_ok=True)

    # generate and save data
    df = sem.generate_data(num_sample)
    df.to_csv(os.path.join(dir, 'data.csv'), index=False)

    # save edges (for evaluation)
    edges = sem.E
    txt_name = os.path.join(dir, 'edges.txt')
    with open(txt_name, 'w') as file:
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]}\n")
    
    print("dataset with {} variables, {} edges, {} samples saved to {}".format(num_var, num_edge, num_sample, dir))

# adj_matrix = np.zeros((num_var, num_var), dtype=int)
# for (u_str, v_str) in edges:
#     u = int(u_str.split('_')[1])
#     v = int(v_str.split('_')[1])
#     adj_matrix[u-1][v-1] = 1
# np.save('adj.npy', adj_matrix)
# showBN(edges,'graph')
# dir = '../datasets/'
# df = sem.generate_data(num_sample)
# filename = dir+'data2_{}_{}.csv'.format(num_var, num_sample)
# df.to_csv(filename, index=False)
# for i in range(8):
#     df = sem.generate_data(num_sample)
#     filename = dir+'data2_{}_{}.csv'.format(num_var, num_sample)
#     df.to_csv(filename, index=False)
#     num_sample *= 2