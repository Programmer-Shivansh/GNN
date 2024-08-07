import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Coordinates of the nodes (index starts from 0)
nodes = {
    0: (4234.93, 4513.14), 1: (6939.72, 7557.96), 2: (8531.68, 9644.51),
    3: (200.93, 3508.50), 4: (1267.39, 4497.68), 5: (1916.54, 5146.83),
    6: (2720.25, 5780.53), 7: (3400.31, 5224.11), 8: (4806.80, 8114.37),
    9: (5363.21, 7635.24), 10: (5919.63, 8315.30), 11: (5301.39, 8639.88),
    12: (6460.59, 6429.68), 13: (6429.68, 4559.51), 14: (6707.88, 9211.75),
    15: (5734.16, 9907.26), 16: (7573.42, 10154.56), 17: (9026.28, 6491.50),
    18: (10540.96, 6491.50), 19: (2812.98, 4636.79), 20: (2009.27, 3848.53),
    21: (7619.78, 6429.68), 22: (7697.06, 4574.96), 23: (9814.53, 4930.45),
    24: (4095.83, 8825.35), 25: (4173.11, 7387.94), 26: (4729.52, 6939.72),
    27: (3384.85, 8129.83), 28: (3400.31, 6615.15), 29: (4018.55, 6151.47)
}

# Normalizing coordinates
coords = np.array(list(nodes.values()))
x = coords[:, 0]
y = coords[:, 1]

x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

# Create a graph and add nodes with normalized positions
G = nx.Graph()
for node_id in nodes.keys():
    G.add_node(node_id, pos=(x_normalized[node_id], y_normalized[node_id]))

# Define edges with attributes
edges = [
    (0, 3, 415.44, 203.20), (0, 7, 46.02, 101.60), (1, 2, 270.66, 152.40), 
    (1, 10, 47.24, 101.60), (1, 12, 94.18, 101.60), (2, 16, 213.06, 25.40), 
    (2, 17, 350.83, 152.40), (3, 4, 334.67, 152.40), (4, 5, 176.17, 152.40),
    (4, 20, 186.23, 152.40), (5, 6, 28.96, 152.40), (6, 7, 127.71, 76.20),
    (6, 28, 63.40, 152.40), (7, 29, 33.53, 50.80), (8, 9, 137.47, 152.40),
    (8, 11, 17.98, 50.80), (8, 24, 126.80, 152.40), (10, 14, 92.35, 304.80),
    (11, 10, 250.85, 25.40), (12, 13, 233.48, 203.20), (13, 22, 116.43, 25.40),
    (14, 15, 231.04, 254.00), (14, 16, 307.54, 25.40), (17, 18, 124.36, 152.40),
    (17, 23, 359.97, 25.40), (20, 19, 34.44, 25.40), (21, 12, 213.67, 152.40),
    (22, 21, 106.99, 152.40), (23, 18, 294.74, 152.40), (25, 8, 82.60, 76.20),
    (25, 26, 96.62, 152.40), (25, 27, 129.24, 152.40), (28, 25, 222.50, 152.40)
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data
elevation = {
    0: 652.58, 1: 649.84, 2: 646.48, 3: 656.39, 4: 652.73, 5: 648.92,
    6: 648.31, 7: 648.31, 8: 642.98, 9: 646.48, 10: 651.97, 11: 643.13,
    12: 651.21, 13: 653.19, 14: 653.49, 15: 655.02, 16: 642.83, 17: 653.49,
    18: 655.17, 19: 652.27, 20: 652.73, 21: 653.49, 22: 657.30, 23: 663.86,
    24: 645.57, 25: 639.93, 26: 640.69, 27: 639.62, 28: 646.18, 29: 647.09
}

# Normalize elevations for color mapping
elevations = np.array(list(elevation.values()))
norm = Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
cmap = plt.colormaps['viridis']  # Updated to use plt.colormaps

# Create a color map based on normalized elevations
node_colors = [cmap(norm(elevation[node])) for node in G.nodes]

# Extract positions
pos = nx.get_node_attributes(G, 'pos')

# # Plotting
# fig, ax = plt.subplots(figsize=(12, 7))
# nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=cmap, font_size=10, font_weight='bold', edge_color='gray', ax=ax)

# # Draw edge labels (length and diameter)
# edge_labels = nx.get_edge_attributes(G, 'length')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_color='red', ax=ax)

# # Draw diameter labels
# diameter_labels = nx.get_edge_attributes(G, 'diameter')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=diameter_labels, label_pos=0.3, font_color='blue', ax=ax)

# # Add colorbar for elevation
# sm = ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, label='Elevation')

# plt.title('Node Graph with Elevation-Based Colors and Edge Annotations')
# plt.show()
