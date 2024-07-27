import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
nodes = {
    0: (8000.00, 0.00), 1: (6000.00, 2000.00), 2: (6000.00, 4000.00),
    3: (7500.00, 4000.00), 4: (9000.00, 4000.00), 5: (10000.00, 4000.00),
    6: (10000.00, 6000.00), 7: (10000.00, 8000.00), 8: (10000.00, 10000.00),
    9: (9000.00, 10000.00), 10: (9000.00, 11500.00), 11: (9000.00, 13000.00),
    12: (7000.00, 13000.00), 13: (8000.00, 10000.00), 14: (7000.00, 10000.00),
    15: (6000.00, 10000.00), 16: (6000.00, 8500.00), 17: (6000.00, 7000.00),
    18: (6000.00, 5500.00), 19: (4500.00, 4000.00), 20: (4500.00, 2000.00),
    21: (4500.00, 0.00), 22: (3000.00, 4000.00), 23: (3000.00, 7000.00),
    24: (3000.00, 10000.00), 25: (4000.00, 10000.00), 26: (5000.00, 10000.00),
    27: (1500.00, 4000.00), 28: (0.00, 4000.00), 29: (0.00, 7000.00),
    30: (0.00, 10000.00), 31: (1500.00, 10000.00)
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
    (0, 1, 100.00, 1016.00), (1, 2, 1350.00, 1016.00), (2, 3, 900.00, 1016.00),
    (3, 4, 1150.00, 1016.00), (4, 5, 1450.00, 1016.00), (5, 6, 450.00, 1016.00),
    (6, 7, 850.00, 1016.00), (7, 8, 850.00, 1016.00), (8, 9, 800.00, 1016.00),
    (9, 10, 950.00, 762.00), (10, 11, 1200.00, 609.60), (11, 12, 3500.00, 609.60),
    (9, 13, 800.00, 508.00), (13, 14, 500.00, 304.80), (14, 15, 550.00, 304.80),
    (15, 16, 2730.00, 304.80), (16, 17, 1750.00, 508.00), (17, 18, 800.00, 508.00),
    (18, 2, 400.00, 609.60), (2, 19, 2200.00, 1016.00), (19, 20, 1500.00, 508.00),
    (20, 21, 500.00, 304.80), (19, 22, 2650.00, 1016.00), (22, 23, 1230.00, 762.00),
    (23, 24, 1300.00, 762.00), (24, 25, 850.00, 508.00), (25, 26, 300.00, 304.80),
    (26, 15, 750.00, 304.80), (22, 27, 1500.00, 406.40), (27, 28, 2000.00, 304.80),
    (28, 29, 1600.00, 304.80), (29, 30, 150.00, 508.00), (30, 31, 860.00, 406.40),
    (31, 24, 950.00, 609.60)
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation data
elevation = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00,
    6: 0.00, 7: 0.00, 8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00,
    12: 0.00, 13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00, 17: 0.00,
    18: 0.00, 19: 0.00, 20: 0.00, 21: 0.00, 22: 0.00, 23: 0.00,
    24: 0.00, 25: 0.00, 26: 0.00, 27: 0.00, 28: 0.00, 29: 0.00,
    30: 0.00, 31: 0.00
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
# plt.show()
