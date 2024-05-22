import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import kneighbors_graph

# Define the distance matrix
dist_matrix = np.array([
    [0, 22.5, 1, 21.25],
    [22.5, 0, 1, 0.25],
    [1, 1, 0, 0],
    [21.25, 0.25, 0, 0]
])

# Compute the k-nearest neighbors graph (k=2)
k = 2
knn_graph = kneighbors_graph(dist_matrix, k, mode='distance', include_self=False)

# Create a graph from the k-nearest neighbors matrix
g_knn = nx.Graph(knn_graph)

# Draw the k-nearest neighbors graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(g_knn)  # positions for all nodes
nx.draw(g_knn, pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=3000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(g_knn, 'weight')
nx.draw_networkx_edge_labels(g_knn, pos, edge_labels=edge_labels, font_size=10)
plt.title(f'K-nearest Neighbors Graph (k={k})')

# Save the plot as a PNG file
plt.savefig("knn_graph.png")

# Optionally, show the plot
plt.show()
