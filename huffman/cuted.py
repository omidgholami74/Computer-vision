import matplotlib.pyplot as plt
import networkx as nx
from numpy import genfromtxt
import numpy as np
import pandas as pd



def normalized_min_cut(graph):
    """Clusters graph nodes according to normalized minimum cut algorithm.
    All nodes must have at least 1 edge. Uses zero as decision boundary. 
    
    Parameters
    -----------
        graph: a networkx graph to cluster
        
    Returns
    -----------
        vector containing -1 or 1 for every node
    References
    ----------
        J. Shi and J. Malik, *Normalized Cuts and Image Segmentation*, 
        IEEE Transactions on Pattern Analysis and Machine Learning, vol. 22, pp. 888-905
    """
    m_adjacency = np.array(nx.to_numpy_matrix(graph))

    D = np.diag(np.sum(m_adjacency, 0))
    D_half_inv = np.diag(1.0 / np.sqrt(np.sum(m_adjacency, 0)))
    M = np.dot(D_half_inv, np.dot((D - m_adjacency), D_half_inv))

    (w, v) = np.linalg.eig(M)
    #find index of second smallest eigenvalue
    index = np.argsort(w)[1]

    v_partition = v[:, index]
    v_partition = np.sign(v_partition)
    return v_partition

input_data = pd.read_csv('matrix.csv', index_col=0)
G = nx.DiGraph(input_data.values)
# nx.draw(G)
# layout = nx.spring_layout(G)
# plt.show() 

v_partition = normalized_min_cut(G)
colors = np.zeros((len(v_partition), 3)) + 1.0
colors[:, 2] = np.where(v_partition >= 0, 1.0, 0)
nx.draw(G, node_color=colors)
plt.show() 


