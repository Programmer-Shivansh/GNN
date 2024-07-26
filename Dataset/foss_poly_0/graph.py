import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
# Node data (updated indices to start from 0)
nodes = {
    0: (7111.65, 7532.36),
    1: (5679.61, 9538.83),
    2: (4862.46, 9538.83),
    3: (2750.81, 9474.11),
    4: (1852.75, 8357.61),
    5: (1974.11, 6076.05),
    6: (1974.11, 5149.68),
    7: (4235.44, 5076.86),
    8: (6411.81, 5093.04),
    9: (5412.62, 7888.35),
    10: (4510.52, 8264.56),
    11: (3033.98, 9243.53),
    12: (2301.78, 8078.48),
    13: (2944.98, 7669.90),
    14: (3786.41, 7139.97),
    15: (4830.10, 6480.58),
    16: (7099.51, 8438.51),
    17: (5505.66, 8450.65),
    18: (3563.92, 8839.00),
    19: (3167.48, 7532.36),
    20: (2730.58, 7285.60),
    21: (3511.33, 6666.67),
    22: (4097.90, 6286.41),
    23: (3337.38, 5121.36),
    24: (4530.74, 6011.33),
    25: (4215.21, 7783.17),
    26: (5194.17, 7055.02),
    27: (5218.45, 5089.00),
    28: (5622.98, 5999.19),
    29: (5950.65, 5796.93),
    30: (6614.08, 7621.36),
    31: (5380.26, 7544.50),
    32: (6318.77, 7281.55),
    33: (6549.35, 7212.78),
    34: (6585.76, 6092.23),
    35: (7152.10, 6104.37),
    36: (7669.90, 7783.17)
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

# Define edges with attributes (updated indices to start from 0)
edges = [
    (0, 16, 132.76, 60.00),
    (16, 1, 374.68, 60.00),
    (1, 2, 119.74, 60.00),
    (2, 3, 312.72, 60.00),
    (3, 4, 289.09, 60.00),
    (4, 5, 336.33, 60.00),
    (5, 6, 135.81, 60.00),
    (6, 23, 201.26, 60.00),
    (23, 7, 132.53, 60.00),
    (7, 27, 144.66, 60.00),
    (27, 8, 175.72, 60.00),
    (8, 35, 112.17, 60.00),
    (35, 0, 210.74, 60.00),
    (0, 30, 75.41, 200.00),
    (30, 9, 181.42, 150.00),
    (9, 10, 146.96, 125.00),
    (10, 18, 162.69, 80.00),
    (18, 11, 99.64, 60.00),
    (11, 3, 52.98, 60.00),
    (1, 17, 162.97, 60.00),
    (17, 9, 83.96, 60.00),
    (9, 31, 49.82, 60.00),
    (31, 26, 78.50, 80.00),
    (26, 15, 99.27, 60.00),
    (15, 24, 82.29, 60.00),
    (24, 7, 147.49, 60.00),
    (2, 10, 197.32, 60.00),
    (10, 25, 83.30, 80.00),
    (25, 14, 113.80, 60.00),
    (14, 21, 80.82, 60.00),
    (21, 6, 340.97, 60.00),
    (4, 12, 77.39, 60.00),
    (12, 13, 112.37, 60.00),
    (13, 19, 37.34, 60.00),
    (19, 14, 108.85, 60.00),
    (14, 15, 182.82, 60.00),
    (15, 28, 136.02, 60.00),
    (28, 29, 56.70, 60.00),
    (29, 8, 124.08, 60.00),
    (16, 17, 234.60, 60.00),
    (11, 12, 203.83, 60.00),
    (18, 19, 248.05, 60.00),
    (13, 20, 65.19, 60.00),
    (20, 5, 210.09, 60.00),
    (20, 21, 147.57, 60.00),
    (21, 22, 103.80, 60.00),
    (23, 22, 210.95, 60.00),
    (22, 24, 75.08, 60.00),
    (25, 26, 180.29, 60.00),
    (27, 28, 149.05, 60.00),
    (28, 32, 215.05, 60.00),
    (31, 32, 144.44, 80.00),
    (32, 33, 34.74, 125.00),
    (30, 33, 59.93, 125.00),
    (33, 34, 165.67, 60.00),
    (29, 34, 119.97, 60.00),
    (34, 35, 83.17, 60.00),
    (36, 0, 1.00, 250.00)
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation data (updated indices to start from 0)
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
    20: 61.75,
    21: 60.90,
    22: 62.20,
    23: 63.80,
    24: 64.00,
    25: 63.90,
    26: 65.30,
    27: 64.50,
    28: 62.80,
    29: 60.00,
    30: 61.50,
    31: 63.00,
    32: 64.20,
    33: 62.50,
    34: 63.60,
    35: 65.70,
    36: 64.10
}

# Normalize elevation data
elevation_values = np.array(list(elevation.values()))
norm = Normalize(vmin=np.min(elevation_values), vmax=np.max(elevation_values))
# cmap = get_cmap('viridis')
# sm = ScalarMappable(cmap=cmap, norm=norm)

# Plot the graph
pos = nx.get_node_attributes(G, 'pos')
# node_colors = [sm.to_rgba(elevation[node]) for node in G.nodes()]
edge_labels = nx.get_edge_attributes(G, 'length')
edge_labels_diameter = nx.get_edge_attributes(G, 'diameter')

# plt.figure(figsize=(12, 12))
# nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_color='white', edge_color='gray')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=8, font_color='black')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_diameter, label_pos=0.7, font_size=8, font_color='blue')

# # Add colorbar
# plt.colorbar(sm, label='Elevation')

plt.title('Graph Visualization with Node Elevation and Edge Attributes')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable, get_cmap

# # Coordinates of the nodes
# # Node data
# nodes = {
#     1: (7111.65, 7532.36),
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
#     37: (7669.90, 7783.17)
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
#     (1, 17, 132.76, 60.00),
#     (17, 2, 374.68, 60.00),
#     (2, 3, 119.74, 60.00),
#     (3, 4, 312.72, 60.00),
#     (4, 5, 289.09, 60.00),
#     (5, 6, 336.33, 60.00),
#     (6, 7, 135.81, 60.00),
#     (7, 24, 201.26, 60.00),
#     (24, 8, 132.53, 60.00),
#     (8, 28, 144.66, 60.00),
#     (28, 9, 175.72, 60.00),
#     (9, 36, 112.17, 60.00),
#     (36, 1, 210.74, 60.00),
#     (1, 31, 75.41, 200.00),
#     (31, 10, 181.42, 150.00),
#     (10, 11, 146.96, 125.00),
#     (11, 19, 162.69, 80.00),
#     (19, 12, 99.64, 60.00),
#     (12, 4, 52.98, 60.00),
#     (2, 18, 162.97, 60.00),
#     (18, 10, 83.96, 60.00),
#     (10, 32, 49.82, 60.00),
#     (32, 27, 78.50, 80.00),
#     (27, 16, 99.27, 60.00),
#     (16, 25, 82.29, 60.00),
#     (25, 8, 147.49, 60.00),
#     (3, 11, 197.32, 60.00),
#     (11, 26, 83.30, 80.00),
#     (26, 15, 113.80, 60.00),
#     (15, 22, 80.82, 60.00),
#     (22, 7, 340.97, 60.00),
#     (5, 13, 77.39, 60.00),
#     (13, 14, 112.37, 60.00),
#     (14, 20, 37.34, 60.00),
#     (20, 15, 108.85, 60.00),
#     (15, 16, 182.82, 60.00),
#     (16, 29, 136.02, 60.00),
#     (29, 30, 56.70, 60.00),
#     (30, 9, 124.08, 60.00),
#     (17, 18, 234.60, 60.00),
#     (12, 13, 203.83, 60.00),
#     (19, 20, 248.05, 60.00),
#     (14, 21, 65.19, 60.00),
#     (21, 6, 210.09, 60.00),
#     (21, 22, 147.57, 60.00),
#     (22, 23, 103.80, 60.00),
#     (24, 23, 210.95, 60.00),
#     (23, 25, 75.08, 60.00),
#     (26, 27, 180.29, 60.00),
#     (28, 29, 149.05, 60.00),
#     (29, 33, 215.05, 60.00),
#     (32, 33, 144.44, 80.00),
#     (33, 34, 34.74, 125.00),
#     (31, 34, 59.93, 125.00),
#     (34, 35, 165.67, 60.00),
#     (30, 35, 119.97, 60.00),
#     (35, 36, 83.17, 60.00),
#     (37, 1, 1.00, 250.00)
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
#     37: 66.50
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

