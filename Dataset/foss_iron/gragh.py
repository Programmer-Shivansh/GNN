# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable

# # Coordinates of the nodes
# nodes = {
#     0: (7111.65, 7532.36), 1: (5679.61, 9538.83), 2: (6772.93, 8303.33),
#     3: (7623.35, 5428.91), 4: (7158.27, 6761.78), 5: (5185.12, 5435.83),
#     6: (6894.72, 4383.77), 7: (5914.82, 5723.63), 8: (6062.34, 8164.20),
#     9: (6684.64, 5958.84), 10: (7261.95, 7491.69), 11: (5140.83, 8328.72),
#     12: (4748.50, 4916.50), 13: (6346.91, 4317.97), 14: (6821.51, 7890.52),
#     15: (5483.79, 8924.30), 16: (8447.54, 7311.90), 17: (7778.64, 4516.95),
#     18: (8099.54, 6588.78), 19: (5028.96, 6073.63), 20: (5454.14, 4164.75),
#     21: (6790.53, 5421.57), 22: (8491.12, 6677.87), 23: (5858.25, 7278.38),
#     24: (5194.31, 7703.88), 25: (5941.85, 7657.55), 26: (6224.30, 8396.99),
#     27: (6232.52, 6926.66), 28: (6395.45, 6301.67), 29: (6545.35, 5712.50)
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
#     G.add_node(node_id, pos=(x_normalized[node_id], y_normalized[node_id]))

# # Define edges with attributes
# edges = [
#     (0, 16, 86.82, 60.00), (0, 6, 81.18, 60.00), (1, 16, 171.10, 130.00),
#     (1, 12, 143.10, 60.00), (1, 21, 162.64, 130.00), (2, 14, 123.94, 60.00),
#     (2, 21, 152.64, 130.00), (3, 4, 163.37, 60.00), (4, 6, 54.38, 130.00),
#     (4, 20, 98.56, 60.00), (5, 7, 68.26, 60.00), (6, 7, 52.21, 60.00),
#     (6, 28, 95.53, 130.00), (7, 29, 76.87, 60.00), (8, 9, 75.66, 60.00),
#     (8, 11, 96.78, 130.00), (8, 24, 99.13, 130.00), (9, 11, 112.72, 60.00),
#     (11, 12, 74.54, 60.00), (11, 14, 131.78, 130.00), (12, 22, 104.52, 60.00),
#     (14, 15, 121.50, 60.00), (14, 16, 70.62, 130.00), (17, 18, 97.48, 60.00),
#     (17, 23, 104.81, 60.00), (20, 19, 105.78, 60.00), (21, 12, 71.73, 60.00),
#     (22, 21, 103.37, 130.00), (23, 18, 85.34, 130.00), (25, 8, 107.77, 60.00),
#     (25, 26, 63.11, 130.00), (25, 27, 66.32, 130.00), (28, 25, 101.42, 60.00)
# ]

# # Add edges with attributes to the graph
# for (node1, node2, length, diameter) in edges:
#     G.add_edge(node1, node2, length=length, diameter=diameter)

# # Elevation and demand data
# elevation = {
#     0: 652.58, 1: 649.84, 2: 646.48, 3: 656.39, 4: 652.73, 5: 648.92,
#     6: 648.31, 7: 648.31, 8: 642.98, 9: 646.48, 10: 651.97, 11: 643.13,
#     12: 651.21, 13: 653.19, 14: 653.49, 15: 655.02, 16: 642.83, 17: 653.49,
#     18: 655.17, 19: 652.27, 20: 652.73, 21: 653.49, 22: 657.30, 23: 663.86,
#     24: 645.57, 25: 639.93, 26: 640.69, 27: 639.62, 28: 646.18, 29: 647.09
# }

# # Normalize elevations for color mapping
# elevations = np.array(list(elevation.values()))
# norm = Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
# cmap = plt.get_cmap('viridis')  # Updated to use plt.get_cmap

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

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
nodes = {
    1: (7111.65, 7532.36), 2: (5679.61, 9538.83), 3: (6772.93, 8303.33),
    4: (7623.35, 5428.91), 5: (7158.27, 6761.78), 6: (5185.12, 5435.83),
    7: (6894.72, 4383.77), 8: (5914.82, 5723.63), 9: (6062.34, 8164.20),
    10: (6684.64, 5958.84), 11: (7261.95, 7491.69), 12: (5140.83, 8328.72),
    13: (4748.50, 4916.50), 14: (6346.91, 4317.97), 15: (6821.51, 7890.52),
    16: (5483.79, 8924.30), 17: (8447.54, 7311.90), 18: (7778.64, 4516.95),
    19: (8099.54, 6588.78), 20: (5028.96, 6073.63), 21: (5454.14, 4164.75),
    22: (6790.53, 5421.57), 23: (8491.12, 6677.87), 24: (5858.25, 7278.38),
    25: (5194.31, 7703.88), 26: (5941.85, 7657.55), 27: (6224.30, 8396.99),
    28: (6232.52, 6926.66), 29: (6395.45, 6301.67), 30: (6545.35, 5712.50)
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
    (1, 17, 86.82, 60.00), (1, 7, 81.18, 60.00), (2, 17, 171.10, 130.00),
    (2, 13, 143.10, 60.00), (2, 22, 162.64, 130.00), (3, 15, 123.94, 60.00),
    (3, 22, 152.64, 130.00), (4, 5, 163.37, 60.00), (5, 7, 54.38, 130.00),
    (5, 21, 98.56, 60.00), (6, 8, 68.26, 60.00), (7, 8, 52.21, 60.00),
    (7, 29, 95.53, 130.00), (8, 30, 76.87, 60.00), (9, 10, 75.66, 60.00),
    (9, 12, 96.78, 130.00), (9, 25, 99.13, 130.00), (10, 12, 112.72, 60.00),
    (12, 13, 74.54, 60.00), (12, 15, 131.78, 130.00), (13, 23, 104.52, 60.00),
    (15, 16, 121.50, 60.00), (15, 17, 70.62, 130.00), (18, 19, 97.48, 60.00),
    (18, 24, 104.81, 60.00), (21, 20, 105.78, 60.00), (22, 13, 71.73, 60.00),
    (23, 22, 103.37, 130.00), (24, 19, 85.34, 130.00), (26, 9, 107.77, 60.00),
    (26, 27, 63.11, 130.00), (26, 28, 66.32, 130.00), (29, 26, 101.42, 60.00)
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
cmap = plt.get_cmap('viridis')  # Updated to use plt.get_cmap

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
