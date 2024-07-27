
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
nodes = {
    1: (4234.93, 4513.14), 2: (6939.72, 7557.96), 3: (8531.68, 9644.51),
    4: (200.93, 3508.50), 5: (1267.39, 4497.68), 6: (1916.54, 5146.83),
    7: (2720.25, 5780.53), 8: (3400.31, 5224.11), 9: (4806.80, 8114.37),
    10: (5363.21, 7635.24), 11: (5919.63, 8315.30), 12: (5301.39, 8639.88),
    13: (6460.59, 6429.68), 14: (6429.68, 4559.51), 15: (6707.88, 9211.75),
    16: (5734.16, 9907.26), 17: (7573.42, 10154.56), 18: (9026.28, 6491.50),
    19: (10540.96, 6491.50), 20: (2812.98, 4636.79), 21: (2009.27, 3848.53),
    22: (7619.78, 6429.68), 23: (7697.06, 4574.96), 24: (9814.53, 4930.45),
    25: (4095.83, 8825.35), 26: (4173.11, 7387.94), 27: (4729.52, 6939.72),
    28: (3384.85, 8129.83), 29: (3400.31, 6615.15), 30: (4018.55, 6151.47)
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
    G.add_node(node_id, pos=(x_normalized[node_id-1], y_normalized[node_id-1]))

# Define edges with attributes
edges = [
    (1, 4, 415.44, 203.20), (1, 8, 46.02, 101.60), (2, 3, 270.66, 152.40), 
    (2, 11, 47.24, 101.60), (2, 13, 94.18, 101.60), (3, 17, 213.06, 25.40), 
    (3, 18, 350.83, 152.40), (4, 5, 334.67, 152.40), (5, 6, 176.17, 152.40),
    (5, 21, 186.23, 152.40), (6, 7, 28.96, 152.40), (7, 8, 127.71, 76.20),
    (7, 29, 63.40, 152.40), (8, 30, 33.53, 50.80), (9, 10, 137.47, 152.40),
    (9, 12, 17.98, 50.80), (9, 25, 126.80, 152.40), (11, 15, 92.35, 304.80),
    (12, 11, 250.85, 25.40), (13, 14, 233.48, 203.20), (14, 23, 116.43, 25.40),
    (15, 16, 231.04, 254.00), (15, 17, 307.54, 25.40), (18, 19, 124.36, 152.40),
    (18, 24, 359.97, 25.40), (21, 20, 34.44, 25.40), (22, 13, 213.67, 152.40),
    (23, 22, 106.99, 152.40), (24, 19, 294.74, 152.40), (26, 9, 82.60, 76.20),
    (26, 27, 96.62, 152.40), (26, 28, 129.24, 152.40), (29, 26, 222.50, 152.40)
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data
elevation = {
    1: 652.58, 2: 649.84, 3: 646.48, 4: 656.39, 5: 652.73, 6: 648.92,
    7: 648.31, 8: 648.31, 9: 642.98, 10: 646.48, 11: 651.97, 12: 643.13,
    13: 651.21, 14: 653.19, 15: 653.49, 16: 655.02, 17: 642.83, 18: 653.49,
    19: 655.17, 20: 652.27, 21: 652.73, 22: 653.49, 23: 657.30, 24: 663.86,
    25: 645.57, 26: 639.93, 27: 640.69, 28: 639.62, 29: 646.18, 30: 647.09
}

# Normalize elevations for color mapping
elevations = np.array(list(elevation.values()))
norm = Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
cmap = plt.colormaps['viridis']  # Updated to use plt.colormaps

# Create a color map based on normalized elevations
node_colors = [cmap(norm(elevation[node])) for node in G.nodes]

# Extract positions
pos = nx.get_node_attributes(G, 'pos')

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))
nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=cmap, font_size=10, font_weight='bold', edge_color='gray', ax=ax)

# Draw edge labels (length and diameter)
edge_labels = nx.get_edge_attributes(G, 'length')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_color='red', ax=ax)

# Draw diameter labels
diameter_labels = nx.get_edge_attributes(G, 'diameter')
nx.draw_networkx_edge_labels(G, pos, edge_labels=diameter_labels, label_pos=0.3, font_color='blue', ax=ax)

# Add colorbar for elevation
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Elevation')

plt.title('Node Graph with Elevation-Based Colors and Edge Annotations')

plt.show()
