
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
nodes = {
    1: (4100.49, 9189.63), 2: (3306.32, 8136.14), 3: (2998.38, 7066.45),
    4: (2868.72, 6110.21), 5: (3030.79, 5121.56), 6: (3241.49, 4230.15),
    7: (3711.51, 3419.77), 8: (4230.15, 2852.51), 9: (4700.16, 2495.95),
    10: (4294.98, 1961.10), 11: (5818.48, 4294.98), 12: (5802.27, 5640.19),
    13: (5769.85, 6207.46), 14: (5461.91, 7017.83), 15: (5089.14, 8282.01),
    16: (6693.68, 810.37), 17: (3354.94, 745.54), 18: (7277.15, 5623.99),
    19: (8687.20, 5931.93), 20: (7730.96, 2933.55)
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
     (1, 2, 3535.69, 4572.00), (2, 3, 6035.06, 4572.00), (3, 4, 2225.05, 4572.00),
    (4, 5, 2529.85, 4572.00), (5, 6, 2621.29, 4572.00), (6, 7, 5821.70, 4572.00),
    (7, 8, 2926.09, 3353.00), (8, 9, 3810.01, 3353.00), (9, 10, 2926.09, 4572.00),
    (10, 11, 3413.77, 5182.00), (11, 12, 4419.61, 5182.00), (12, 13, 3718.57, 5182.00),
    (13, 14, 7345.70, 5182.00), (14, 15, 6431.30, 5182.00), (15, 1, 4724.42, 5182.00),
    (16, 10, 8046.75, 1829.00), (17, 12, 9509.79, 1829.00), (18, 18, 7315.22, 1524.00),
    (19, 11, 4389.13, 1524.00), (20, 20, 11704.36, 1524.00), 
    (15, 1, 4724.42, 2743.21), (16, 10, 8046.75, 2438.41), (17, 12, 9509.79, 2438.41),
    (18, 18, 7315.22, 2133.61), (19, 11, 4389.13, 1828.81), (20, 9, 8046.75, 1828.81)   
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data
elevation = {
    2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00, 6: 0.00, 7: 0.00,
    8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00, 12: 0.00, 13: 0.00,
    14: 0.00, 15: 0.00, 16: 0.00, 17: 0.00, 18: 0.00, 19: 0.00,
    20: 0.00, 1: 0.00
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
