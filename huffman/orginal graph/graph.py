import matplotlib.pyplot as plt
import networkx as nx
from numpy import genfromtxt
import numpy as np
import pandas as pd



input_data = pd.read_csv('matrix.csv', index_col=0)
G = nx.DiGraph(input_data.values)
nx.draw(G)
plt.show() 

