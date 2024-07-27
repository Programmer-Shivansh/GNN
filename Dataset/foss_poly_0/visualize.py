
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
# Node data
nodes = {
    1: (7111.65, 7532.36),
    2: (5679.61, 9538.83),
    3: (4862.46, 9538.83),
    4: (2750.81, 9474.11),
    5: (1852.75, 8357.61),
    6: (1974.11, 6076.05),
    7: (1974.11, 5149.68),
    8: (4235.44, 5076.86),
    9: (6411.81, 5093.04),
    10: (5412.62, 7888.35),
    11: (4510.52, 8264.56),
    12: (3033.98, 9243.53),
    13: (2301.78, 8078.48),
    14: (2944.98, 7669.90),
    15: (3786.41, 7139.97),
    16: (4830.10, 6480.58),
    17: (7099.51, 8438.51),
    18: (5505.66, 8450.65),
    19: (3563.92, 8839.00),
    20: (3167.48, 7532.36),
    21: (2730.58, 7285.60),
    22: (3511.33, 6666.67),
    23: (4097.90, 6286.41),
    24: (3337.38, 5121.36),
    25: (4530.74, 6011.33),
    26: (4215.21, 7783.17),
    27: (5194.17, 7055.02),
    28: (5218.45, 5089.00),
    29: (5622.98, 5999.19),
    30: (5950.65, 5796.93),
    31: (6614.08, 7621.36),
    32: (5380.26, 7544.50),
    33: (6318.77, 7281.55),
    34: (6549.35, 7212.78),
    35: (6585.76, 6092.23),
    36: (7152.10, 6104.37),
    37: (7669.90, 7783.17)
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
# Edge data
edges = [
    (1, 17, 132.76, 60.00),
    (17, 2, 374.68, 60.00),
    (2, 3, 119.74, 60.00),
    (3, 4, 312.72, 60.00),
    (4, 5, 289.09, 60.00),
    (5, 6, 336.33, 60.00),
    (6, 7, 135.81, 60.00),
    (7, 24, 201.26, 60.00),
    (24, 8, 132.53, 60.00),
    (8, 28, 144.66, 60.00),
    (28, 9, 175.72, 60.00),
    (9, 36, 112.17, 60.00),
    (36, 1, 210.74, 60.00),
    (1, 31, 75.41, 200.00),
    (31, 10, 181.42, 150.00),
    (10, 11, 146.96, 125.00),
    (11, 19, 162.69, 80.00),
    (19, 12, 99.64, 60.00),
    (12, 4, 52.98, 60.00),
    (2, 18, 162.97, 60.00),
    (18, 10, 83.96, 60.00),
    (10, 32, 49.82, 60.00),
    (32, 27, 78.50, 80.00),
    (27, 16, 99.27, 60.00),
    (16, 25, 82.29, 60.00),
    (25, 8, 147.49, 60.00),
    (3, 11, 197.32, 60.00),
    (11, 26, 83.30, 80.00),
    (26, 15, 113.80, 60.00),
    (15, 22, 80.82, 60.00),
    (22, 7, 340.97, 60.00),
    (5, 13, 77.39, 60.00),
    (13, 14, 112.37, 60.00),
    (14, 20, 37.34, 60.00),
    (20, 15, 108.85, 60.00),
    (15, 16, 182.82, 60.00),
    (16, 29, 136.02, 60.00),
    (29, 30, 56.70, 60.00),
    (30, 9, 124.08, 60.00),
    (17, 18, 234.60, 60.00),
    (12, 13, 203.83, 60.00),
    (19, 20, 248.05, 60.00),
    (14, 21, 65.19, 60.00),
    (21, 6, 210.09, 60.00),
    (21, 22, 147.57, 60.00),
    (22, 23, 103.80, 60.00),
    (24, 23, 210.95, 60.00),
    (23, 25, 75.08, 60.00),
    (26, 27, 180.29, 60.00),
    (28, 29, 149.05, 60.00),
    (29, 33, 215.05, 60.00),
    (32, 33, 144.44, 80.00),
    (33, 34, 34.74, 125.00),
    (31, 34, 59.93, 125.00),
    (34, 35, 165.67, 60.00),
    (30, 35, 119.97, 60.00),
    (35, 36, 83.17, 60.00),
    (37, 1, 1.00, 250.00)
]


# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data
# Elevation data
elevation = {
    1: 65.15,
    2: 64.40,
    3: 63.35,
    4: 62.50,
    5: 61.24,
    6: 65.40,
    7: 67.90,
    8: 66.50,
    9: 66.00,
    10: 64.17,
    11: 63.70,
    12: 62.64,
    13: 61.90,
    14: 62.60,
    15: 63.50,
    16: 64.30,
    17: 65.50,
    18: 64.10,
    19: 62.90,
    20: 62.83,
    21: 62.80,
    22: 63.90,
    23: 64.20,
    24: 67.50,
    25: 64.40,
    26: 63.40,
    27: 63.90,
    28: 65.65,
    29: 64.50,
    30: 64.10,
    31: 64.40,
    32: 64.20,
    33: 64.60,
    34: 64.70,
    35: 65.43,
    36: 65.90,
    37: 66.50
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

