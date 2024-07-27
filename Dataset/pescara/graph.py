import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Coordinates of the nodes
nodes = {
    0: (656582.88, 963296.25),
    1: (657289.06, 963069.50),
    2: (656852.19, 963283.19),
    3: (657294.63, 963262.25),
    4: (657933.44, 963588.94),
    5: (661346.75, 960658.38),
    6: (661252.19, 960607.19),
    7: (661547.63, 960688.13),
    8: (660262.19, 961842.69),
    9: (659947.69, 961952.63),
    10: (659120.75, 962107.19),
    11: (661689.69, 962215.25),
    12: (661623.88, 962717.69),
    13: (662482.81, 962917.75),
    16: (662131.88, 963005.56),
    17: (661006.88, 962760.81),
    18: (660673.44, 962792.06),
    19: (661012.00, 963102.44),
    20: (661221.06, 963474.69),
    21: (661856.94, 963711.13),
    22: (661860.75, 963473.88),
    23: (662326.00, 962957.19),
    24: (662031.81, 962086.69),
    25: (662135.06, 963007.19),
    26: (661853.69, 963051.00),
    27: (661141.50, 961352.25),
    28: (661319.63, 961368.81),
    29: (660878.69, 961712.13),
    30: (660390.13, 962876.63),
    31: (660381.19, 963121.00),
    32: (660557.31, 961781.06),
    33: (660596.63, 962340.81),
    34: (661308.00, 961400.00),
    35: (660857.63, 962345.56),
    36: (660500.44, 962822.94),
    37: (660618.31, 962796.50),
    38: (660651.25, 962646.69),
    39: (660461.31, 962359.19),
    40: (660533.31, 962348.81),
    41: (659046.31, 962535.19),
    43: (659015.00, 962761.75),
    44: (660704.56, 963432.50),
    45: (660447.38, 963119.13),
    46: (659504.06, 963179.19),
    47: (659732.88, 963154.94),
    48: (660377.06, 963121.75),
    49: (660375.88, 963276.44),
    50: (659734.19, 963310.88),
    51: (660371.38, 963483.06),
    52: (659739.94, 963467.50),
    53: (659230.13, 963333.94),
    54: (659229.19, 963481.63),
    55: (659457.44, 963155.31),
    56: (659097.25, 963096.19),
    57: (658198.44, 963134.44),
    58: (658026.94, 963233.88),
    59: (658114.75, 963350.06),
    60: (658119.75, 963536.75),
    61: (658025.81, 963147.50),
    75: (662523.13, 960626.31),
    78: (662528.25, 962839.88),
    79: (662636.56, 960486.38),
    80: (658738.69, 962404.75),
    81: (662454.00, 963567.13),
    82: (662458.63, 963565.25),
    83: (659408.69, 963212.44),
    84: (662548.06, 963787.94),
    85: (659408.69, 963642.06),
    86: (658728.69, 962454.75),
    87: (659408.69, 963325.81),
    88: (659408.69, 963476.63),
    42: (658738.69, 962454.75),
    64: (662568.19, 960492.44),
    14: (662492.56, 962874.81)
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
    (0, 1, 977.36, 100.00),
    (2, 3, 443.17, 150.00),
    (2, 4, 1410.77, 100.00),
    (2, 0, 269.61, 125.00),
    (5, 6, 116.52, 100.00),
    (5, 7, 203.06, 100.00),
    (8, 5, 2309.14, 100.00),
    (9, 10, 1479.98, 150.00),
    (14, 13, 43.97, 350.00),
    (12, 16, 589.16, 200.00),
    (17, 12, 618.53, 200.00),
    (17, 18, 341.10, 200.00),
    (19, 17, 341.92, 100.00),
    (20, 19, 434.24, 100.00),
    (21, 20, 240.59, 100.00),
    (23, 24, 957.44, 200.00),
    (13, 23, 171.04, 350.00),
    (23, 16, 235.57, 300.00),
    (25, 16, 3.59, 200.00),
    (25, 26, 324.36, 200.00),
    (20, 21, 749.74, 100.00),
    (21, 20, 639.70, 100.00),
    (26, 21, 422.95, 100.00),
    (19, 26, 843.52, 125.00),
    (18, 19, 609.23, 100.00),
    (27, 28, 179.04, 150.00),
    (27, 29, 570.90, 100.00),
    (29, 27, 575.73, 100.00),
    (30, 31, 250.28, 100.00),
    (29, 32, 442.11, 100.00),
    (33, 29, 762.35, 100.00),
    (34, 35, 1124.90, 100.00),
    (30, 10, 1903.75, 100.00),
    (36, 30, 125.51, 125.00),
    (36, 37, 121.27, 200.00),
    (18, 37, 55.32, 200.00),
    (39, 38, 424.86, 100.00),
    (39, 40, 72.80, 100.00),
    (42, 41, 186.20, 100.00),
    (43, 44, 477.07, 100.00),
    (43, 45, 384.32, 100.00),
    (46, 45, 478.04, 100.00),
    (47, 46, 206.38, 100.00),
    (50, 49, 265.93, 100.00),
    (51, 50, 197.40, 150.00),
    (52, 51, 150.38, 100.00),
    (53, 52, 122.00, 100.00),
    (54, 53, 158.41, 150.00),
    (55, 54, 91.84, 100.00),
    (56, 55, 158.26, 100.00),
    (57, 56, 150.41, 150.00),
    (58, 57, 174.19, 100.00),
    (59, 58, 126.78, 100.00),
    (60, 59, 206.79, 100.00),
    (61, 60, 225.59, 150.00),
    (75, 77, 348.61, 100.00),
    (78, 79, 486.64, 100.00),
    (80, 81, 111.58, 150.00),
    (83, 84, 417.79, 150.00),
    (84, 85, 209.70, 150.00),
    (86, 87, 237.40, 100.00),
    (87, 88, 144.58, 100.00),
    (42, 86, 420.12, 100.00),
    (64, 42, 487.68, 100.00)
]

# Add edges with attributes to the graph
for u, v, weight, time in edges:
    G.add_edge(u, v, weight=weight, time=time)

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
weights = nx.get_edge_attributes(G, 'weight')
edge_colors = [weights[edge] for edge in G.edges()]

# Create a color map
cmap = get_cmap('viridis')
norm = Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
sm = ScalarMappable(norm=norm, cmap=cmap)

# Draw nodes and edges
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edge_color=[sm.to_rgba(weight) for weight in edge_colors], width=1)
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Create a colorbar for edge weights
plt.colorbar(sm, label='Edge Weight')

plt.title('Network Graph with Normalized Coordinates')
# plt.show()

