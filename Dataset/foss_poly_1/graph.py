import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
# Node data
nodes = {
    0: (5679.61, 9538.83),
    1: (4862.46, 9538.83),
    2: (2750.81, 9474.11),
    3: (1852.75, 8357.61),
    4: (1974.11, 6076.05),
    5: (1974.11, 5149.68),
    6: (4235.44, 5076.86),
    7: (6411.81, 5093.04),
    8: (5412.62, 7888.35),
    9: (4510.52, 8264.56),
    10: (3033.98, 9243.53),
    11: (2301.78, 8078.48),
    12: (2944.98, 7669.90),
    13: (3786.41, 7139.97),
    14: (4830.10, 6480.58),
    15: (7099.51, 8438.51),
    16: (5505.66, 8450.65),
    17: (3563.92, 8839.00),
    18: (3167.48, 7532.36),
    19: (2730.58, 7285.60),
    20: (3511.33, 6666.67),
    21: (4097.90, 6286.41),
    22: (3337.38, 5121.36),
    23: (4530.74, 6011.33),
    24: (4215.21, 7783.17),
    25: (5194.17, 7055.02),
    26: (5218.45, 5089.00),
    27: (5622.98, 5999.19),
    28: (5950.65, 5796.93),
    29: (6614.08, 7621.36),
    30: (5380.26, 7544.50),
    31: (6318.77, 7281.55),
    32: (6549.35, 7212.78),
    33: (6585.76, 6092.23),
    34: (7152.10, 6104.37),
    35: (7111.65, 7532.36),
    36: (7669.90, 7783.17),
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
# Edge data
edges = [
    (0, 16, 132.76, 40.80),
    (16, 1, 374.68, 16.00),
    (1, 2, 119.74, 16.00),
    (2, 3, 312.72, 16.00),
    (3, 4, 289.09, 26.00),
    (4, 5, 336.33, 16.00),
    (5, 6, 135.81, 16.00),
    (6, 21, 201.26, 16.00),
    (21, 7, 132.53, 16.00),
    (7, 26, 144.66, 16.00),
    (26, 8, 175.72, 20.40),
    (8, 34, 112.17, 32.60),
    (34, 0, 210.74, 51.40),
    (0, 30, 75.41, 204.60),
    (30, 9, 181.42, 184.00),
    (9, 10, 146.96, 147.20),
    (10, 18, 162.69, 90.00),
    (18, 11, 99.64, 73.60),
    (11, 3, 52.98, 40.80),
    (1, 17, 162.97, 40.80),
    (17, 9, 83.96, 73.60),
    (9, 31, 49.82, 102.20),
    (31, 25, 78.50, 90.00),
    (25, 14, 99.27, 73.60),
    (14, 22, 82.29, 61.40),
    (22, 7, 147.49, 32.60),
    (3, 9, 197.32, 40.80),
    (9, 24, 83.30, 102.20),
    (24, 15, 113.80, 90.00),
    (15, 20, 80.82, 61.40),
    (20, 5, 340.97, 26.00),
    (4, 12, 77.39, 16.00),
    (12, 13, 112.37, 16.00),
    (13, 19, 37.34, 40.80),
    (19, 15, 108.85, 51.40),
    (15, 14, 182.82, 16.00),
    (14, 27, 136.02, 16.00),
    (27, 28, 56.70, 16.00),
    (28, 7, 124.08, 16.00),
    (16, 17, 234.60, 16.00),
    (11, 12, 203.83, 40.80),
    (18, 19, 248.05, 16.00),
    (13, 20, 65.19, 32.60),
    (20, 4, 210.09, 32.60),
    (20, 21, 147.57, 40.80),
    (21, 22, 103.80, 16.00),
    (23, 22, 210.95, 32.60),
    (22, 23, 75.08, 51.40),
    (24, 25, 180.29, 16.00),
    (26, 27, 149.05, 16.00),
    (27, 32, 215.05, 32.60),
    (32, 33, 144.44, 16.00),
    (33, 34, 34.74, 51.40),
    (30, 33, 59.93, 73.60),
    (33, 35, 165.67, 32.60),
    (28, 35, 119.97, 20.40),
    (35, 34, 83.17, 32.60),
    (36, 0, 1.00, 229.20),
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data
# Elevation data
elevation = {
    0: 65.15,
    1: 64.40,
    2: 63.35,
    3: 62.50,
    4: 61.24,
    5: 65.40,
    6: 67.90,
    7: 66.50,
    8: 66.00,
    9: 64.17,
    10: 63.70,
    11: 62.64,
    12: 61.90,
    13: 62.60,
    14: 63.50,
    15: 64.30,
    16: 65.50,
    17: 64.10,
    18: 62.90,
    19: 62.83,
    20: 62.80,
    21: 65.80,
    22: 67.20,
    23: 68.20,
    24: 69.10,
    25: 68.40,
    26: 66.90,
    27: 66.00,
    28: 64.80,
    29: 64.40,
    30: 63.20,
    31: 62.90,
    32: 63.50,
    33: 65.70,
    34: 66.60,
    35: 68.00,
    36: 70.20,
}

# Demand data
demand = {
    0: 1.00,
    1: 2.00,
    2: 2.50,
    3: 3.00,
    4: 4.00,
    5: 3.50,
    6: 4.50,
    7: 2.50,
    8: 3.00,
    9: 2.00,
    10: 3.00,
    11: 1.50,
    12: 2.50,
    13: 3.50,
    14: 4.00,
    15: 4.50,
    16: 5.00,
    17: 2.50,
    18: 3.00,
    19: 3.50,
    20: 4.00,
    21: 4.50,
    22: 5.00,
    23: 4.00,
    24: 3.50,
    25: 4.50,
    26: 2.50,
    27: 2.00,
    28: 3.00,
    29: 2.50,
    30: 1.00,
    31: 1.50,
    32: 2.00,
    33: 2.50,
    34: 3.00,
    35: 3.50,
    36: 4.00,
}

# Create node colors based on elevation
# elevation_values = np.array([elevation[node] for node in G.nodes()])
# norm = Normalize(vmin=np.min(elevation_values), vmax=np.max(elevation_values))
# cmap = get_cmap('viridis')
# sm = ScalarMappable(norm=norm, cmap=cmap)
# node_colors = [sm.to_rgba(elevation[node]) for node in G.nodes()]

# Create edge widths based on demand
# edge_widths = [demand[node1] / 2 + demand[node2] / 2 for node1, node2 in G.edges()]

# # Draw the graph with normalized node positions
pos = nx.get_node_attributes(G, 'pos')
# plt.figure(figsize=(12, 12))
# nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, edge_color='gray', width=edge_widths)
# plt.colorbar(sm, label='Elevation')

# plt.title("Graph Visualization with Normalized Coordinates")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable, get_cmap

# # Coordinates of the nodes
# # Node data
# nodes = {
#     2: (5679.61, 9538.83),
#     3: (4862.46, 9538.83),
#     4: (2750.81, 9474.11),
#     5: (1852.75, 8357.61),
#     6: (1974.11, 6076.05),
#     7: (1974.11, 5149.68),
#     8: (4235.44, 5076.86),
#     9: (6411.81, 5093.04),
#     10: (5412.62, 7888.35),
#     11: (4510.52, 8264.56),
#     12: (3033.98, 9243.53),
#     13: (2301.78, 8078.48),
#     14: (2944.98, 7669.90),
#     15: (3786.41, 7139.97),
#     16: (4830.10, 6480.58),
#     17: (7099.51, 8438.51),
#     18: (5505.66, 8450.65),
#     19: (3563.92, 8839.00),
#     20: (3167.48, 7532.36),
#     21: (2730.58, 7285.60),
#     22: (3511.33, 6666.67),
#     23: (4097.90, 6286.41),
#     24: (3337.38, 5121.36),
#     25: (4530.74, 6011.33),
#     26: (4215.21, 7783.17),
#     27: (5194.17, 7055.02),
#     28: (5218.45, 5089.00),
#     29: (5622.98, 5999.19),
#     30: (5950.65, 5796.93),
#     31: (6614.08, 7621.36),
#     32: (5380.26, 7544.50),
#     33: (6318.77, 7281.55),
#     34: (6549.35, 7212.78),
#     35: (6585.76, 6092.23),
#     36: (7152.10, 6104.37),
#     1: (7111.65, 7532.36),
#     37: (7669.90, 7783.17),
# }

# # Normalizing coordinates
# coords = np.array(list(nodes.values()))
# x = coords[:, 0]
# y = coords[:, 1]

# x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
# y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

# # Create a graph and add nodes with normalized positions
# G = nx.Graph()
# for node_id in nodes.keys():
#     G.add_node(node_id, pos=(x_normalized[node_id-1], y_normalized[node_id-1]))

# # Define edges with attributes
# # Edge data
# edges = [
#     (1, 17, 132.76, 40.80),
#     (17, 2, 374.68, 16.00),
#     (2, 3, 119.74, 16.00),
#     (3, 4, 312.72, 16.00),
#     (4, 5, 289.09, 26.00),
#     (5, 6, 336.33, 16.00),
#     (6, 7, 135.81, 16.00),
#     (7, 24, 201.26, 16.00),
#     (24, 8, 132.53, 16.00),
#     (8, 28, 144.66, 16.00),
#     (28, 9, 175.72, 20.40),
#     (9, 36, 112.17, 32.60),
#     (36, 1, 210.74, 51.40),
#     (1, 31, 75.41, 204.60),
#     (31, 10, 181.42, 184.00),
#     (10, 11, 146.96, 147.20),
#     (11, 19, 162.69, 90.00),
#     (19, 12, 99.64, 73.60),
#     (12, 4, 52.98, 40.80),
#     (2, 18, 162.97, 40.80),
#     (18, 10, 83.96, 73.60),
#     (10, 32, 49.82, 102.20),
#     (32, 27, 78.50, 90.00),
#     (27, 16, 99.27, 73.60),
#     (16, 25, 82.29, 61.40),
#     (25, 8, 147.49, 32.60),
#     (3, 11, 197.32, 40.80),
#     (11, 26, 83.30, 102.20),
#     (26, 15, 113.80, 90.00),
#     (15, 22, 80.82, 61.40),
#     (22, 7, 340.97, 26.00),
#     (5, 13, 77.39, 16.00),
#     (13, 14, 112.37, 16.00),
#     (14, 20, 37.34, 40.80),
#     (20, 15, 108.85, 51.40),
#     (15, 16, 182.82, 16.00),
#     (16, 29, 136.02, 16.00),
#     (29, 30, 56.70, 16.00),
#     (30, 9, 124.08, 16.00),
#     (17, 18, 234.60, 16.00),
#     (12, 13, 203.83, 40.80),
#     (19, 20, 248.05, 16.00),
#     (14, 21, 65.19, 32.60),
#     (21, 6, 210.09, 32.60),
#     (21, 22, 147.57, 40.80),
#     (22, 23, 103.80, 16.00),
#     (24, 23, 210.95, 32.60),
#     (23, 25, 75.08, 51.40),
#     (26, 27, 180.29, 16.00),
#     (28, 29, 149.05, 16.00),
#     (29, 33, 215.05, 32.60),
#     (32, 33, 144.44, 16.00),
#     (33, 34, 34.74, 51.40),
#     (31, 34, 59.93, 73.60),
#     (34, 35, 165.67, 32.60),
#     (30, 35, 119.97, 20.40),
#     (35, 36, 83.17, 32.60),
#     (37, 1, 1.00, 229.20),
# ]
# # Add edges with attributes to the graph
# for (node1, node2, length, diameter) in edges:
#     G.add_edge(node1, node2, length=length, diameter=diameter)

# # Elevation and demand data
# # Elevation data
# elevation = {
#     1: 65.15,
#     2: 64.40,
#     3: 63.35,
#     4: 62.50,
#     5: 61.24,
#     6: 65.40,
#     7: 67.90,
#     8: 66.50,
#     9: 66.00,
#     10: 64.17,
#     11: 63.70,
#     12: 62.64,
#     13: 61.90,
#     14: 62.60,
#     15: 63.50,
#     16: 64.30,
#     17: 65.50,
#     18: 64.10,
#     19: 62.90,
#     20: 62.83,
#     21: 62.80,
#     22: 63.90,
#     23: 64.20,
#     24: 67.50,
#     25: 64.40,
#     26: 63.40,
#     27: 63.90,
#     28: 65.65,
#     29: 64.50,
#     30: 64.10,
#     31: 64.40,
#     32: 64.20,
#     33: 64.60,
#     34: 64.70,
#     35: 65.43,
#     36: 65.90,
#     37: 66.00
# }
# # Normalize elevations for color mapping
# elevations = np.array(list(elevation.values()))
# norm = Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
# cmap = plt.colormaps['viridis']  # Updated to use plt.colormaps

# # Create a color map based on normalized elevations
# node_colors = [cmap(norm(elevation[node])) for node in G.nodes]

# # Extract positions
# pos = nx.get_node_attributes(G, 'pos')

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
