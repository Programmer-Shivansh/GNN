import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()


G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

G.add_edges_from(
    [
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 7),
        (4, 8),
        (4, 9),
        (5, 7),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 10),
        (7, 11),
        (7, 12),
        (8, 10),
        (8, 11),
        (8, 12),
        (9, 10),
        (9, 11),
        (9, 12),
    ]
)

custom_pos = {
    1: (-1, 1),
    2: (-1, 0),
    3: (-1, -1),
    4: (-0.33, 1),
    5: (-0.33, 0),
    6: (-0.33, -1),
    7: (0.33, 1),
    8: (0.33, 0),
    9: (0.33, -1),
    10: (1, 1),
    11: (1, 0),
    12: (1, -1),
}
spring_pos = nx.spring_layout(G)
pos = {node: custom_pos.get(node, spring_pos[node]) for node in G.nodes()}
nx.draw(G, pos, with_labels=True, node_color="blue", node_size=500, font_size=16)
plt.show()
