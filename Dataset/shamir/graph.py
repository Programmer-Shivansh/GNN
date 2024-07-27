import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes with updated IDs starting from 0
nodes = {
    0: (3000.00, 3000.00),  # Reservoir
    1: (2000.00, 3000.00),  # Demand
    2: (1000.00, 3000.00),  # Demand
    3: (2000.00, 2000.00),  # Demand
    4: (1000.00, 2000.00),  # Demand
    5: (2000.00, 1000.00),  # Demand
    6: (1000.00, 1000.00)   # Demand
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

# Define edges with attributes (updated to use new node IDs)
edges = [
    (0, 1, 1000.00, 457.20),  # (Node1, Node2, Length, Diameter)
    (1, 2, 1000.00, 254.00),
    (1, 3, 1000.00, 406.40),
    (3, 4, 1000.00, 101.60),
    (3, 5, 1000.00, 406.40),
    (5, 6, 1000.00, 254.00),
    (2, 4, 1000.00, 254.00),
    (4, 6, 1000.00, 25.40)
]

# Add edges with attributes to the graph
for (node1, node2, length, diameter) in edges:
    G.add_edge(node1, node2, length=length, diameter=diameter)

# Elevation and demand data with updated IDs
elevation = {
    1: 150.00,
    2: 160.00,
    3: 155.00,
    4: 150.00,
    5: 165.00,
    6: 160.00,
    0: 150.00
}

# Normalize elevations for color mapping
elevations = np.array(list(elevation.values()))
norm = Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
cmap = plt.get_cmap('viridis')

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
