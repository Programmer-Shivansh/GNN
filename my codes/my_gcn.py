import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNWithAttention, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # GAT layers
        self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
        self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Attention layers
        x = self.attn1(x, edge_index)
        x = F.relu(x)
        x = self.attn2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Define your custom graph data
num_nodes = 12
num_edges = 27

# Node features: a matrix with shape [num_nodes, num_node_features]
node_features = torch.tensor(
    [
        [218.75, 0.01341, 0.01768, 0.02211],
        [218.45, 0.00913, 0.01341, 0.01768],
        [218.15, 0.00485, 0.00913, 0.01355],
        [217.81, 0.01671, 0.02282, 0.02914],
        [217.51, 0.01060, 0.01671, 0.02303],
        [217.20, 0.00428, 0.01040, 0.01671],
        [216.99, 0.00118, 0.00523, 0.00915],
        [216.69, 0, 0.00131, 0.00523],
        [216.38, 0, 0, 0.00118],
        [216.90, 0, 0, 0],
        [216.59, 0, 0, 0],
        [216.29, 0, 0, 0],
    ]
)

# Edge indices and custom lengths
edges = {
    (0, 3): 5399.60,
    (0, 4): 4713.72,
    (0, 5): 4841.50,
    (1, 3): 6920.39,
    (1, 4): 5754.89,
    (1, 5): 5069.28,
    (2, 3): 7148.16,
    (2, 4): 7275.94,
    (2, 5): 6110.40,
    (3, 6): 3959.53,
    (3, 7): 3472.11,
    (3, 8): 3561.55,
    (4, 6): 5069.27,
    (4, 7): 4238.42,
    (4, 8): 3750.99,
    (5, 6): 6356.11,
    (5, 7): 5348.16,
    (5, 8): 4517.31,
    (6, 9): 12933.07,
    (6, 10): 9066.66,
    (6, 11): 7523.21,
    (7, 9): 3e20,
    (7, 10): 13311.95,
    (7, 11): 9445.56,
    (8, 9): 3e20,
    (8, 10): 3e20,
    (8, 11): 13690.33,
}
edge_index = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()

# Labels for the nodes (e.g., part of the optimal path or not)
y = torch.zeros(num_nodes, dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index, y=y)

custom_pos = {
    0: (-1, 1),
    1: (-1, 0),
    2: (-1, -1),
    3: (-0.33, 1),
    4: (-0.33, 0),
    5: (-0.33, -1),
    6: (0.33, 1),
    7: (0.33, 0),
    8: (0.33, -1),
    9: (1, 1),
    10: (1, 0),
    11: (1, -1),
}

def create_nx_graph(data, edge_lengths):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(
            edge_index[0, i],
            edge_index[1, i],
            length=edge_lengths[(edge_index[0, i], edge_index[1, i])],
        )
    return G

# Visualize the initial graph and return positions
def visualize_initial_graph(G, pos):
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color="blue"
    )
    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Create and show the initial graph
G = create_nx_graph(data, edges)
visualize_initial_graph(G, custom_pos)

# Get user input for source and target
source = int(input("Enter source node: "))
target = int(input("Enter target node: "))

# Create the model
model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training the model for 100 epochs
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Save the model
torch.save(model.state_dict(), 'gcn_with_attention_model.pth')

# Load the model (for future use)
model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
model.load_state_dict(torch.load('gcn_with_attention_model.pth'))

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        return pred

# Get the predictions
pred = evaluate()
print(f"Predictions: {pred}")

def find_optimal_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="length")
    except nx.NetworkXNoPath:
        path = [source, target]  # If no path exists, return a direct connection (for visualization)
    return path

# Find the optimal path
optimal_path = find_optimal_path(G, source, target)
print(f"Optimal path from {source} to {target}: {optimal_path}")

# Visualization function
def visualize_graph(G, path, pos):
    # Draw nodes
    node_colors = ["red" if node in path else "blue" for node in G.nodes()]
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        cmap=plt.cm.Blues,
    )

    # Highlight the path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

# Visualize the graph with the predicted path using the same layout
visualize_graph(G, optimal_path, custom_pos)



# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


# # Define your custom graph data
# num_nodes = 12
# num_edges = 27

# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     [
#         # elevation , 3 slopes
#         [218.75, 0.01341, 0.01768, 0.02211],
#         [218.45, 0.00913, 0.01341, 0.01768],
#         [218.15, 0.00485, 0.00913, 0.01355],
#         [217.81, 0.01671, 0.02282, 0.02914],
#         [217.51, 0.01060, 0.01671, 0.02303],
#         [217.20, 0.00428, 0.01040, 0.01671],
#         [216.99, 0.00118, 0.00523, 0.00915],
#         [216.69, 0, 0.00131, 0.00523],
#         [216.38, 0, 0, 0.00118],
#         [216.90, 0, 0, 0],
#         [216.59, 0, 0, 0],
#         [216.29, 0, 0, 0],
#     ]
# )

# # Edge indices and custom lengths
# edges = {
#     (0, 3): 5399.60,
#     (0, 4): 4713.72,
#     (0, 5): 4841.50,
#     (1, 3): 6920.39,
#     (1, 4): 5754.89,
#     (1, 5): 5069.28,
#     (2, 3): 7148.16,
#     (2, 4): 7275.94,
#     (2, 5): 6110.40,
#     (3, 6): 3959.53,
#     (3, 7): 3472.11,
#     (3, 8): 3561.55,
#     (4, 6): 5069.27,
#     (4, 7): 4238.42,
#     (4, 8): 3750.99,
#     (5, 6): 6356.11,
#     (5, 7): 5348.16,
#     (5, 8): 4517.31,
#     (6, 9): 12933.07,
#     (6, 10): 9066.66,
#     (6, 11): 7523.21,
#     (7, 9): 3e20,
#     (7, 10): 13311.95,
#     (7, 11): 9445.56,
#     (8, 9): 3e20,
#     (8, 10): 3e20,
#     (8, 11): 13690.33,
# }
# edge_index = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()
# # Labels for the nodes (e.g., part of the optimal path or not)
# y = torch.zeros(num_nodes, dtype=torch.long)
# data = Data(x=node_features, edge_index=edge_index, y=y)

# custom_pos = {
#     0: (-1, 1),
#     1: (-1, 0),
#     2: (-1, -1),
#     3: (-0.33, 1),
#     4: (-0.33, 0),
#     5: (-0.33, -1),
#     6: (0.33, 1),
#     7: (0.33, 0),
#     8: (0.33, -1),
#     9: (1, 1),
#     10: (1, 0),
#     11: (1, -1),
# }


# def create_nx_graph(data, edge_lengths):
#     G = nx.Graph()
#     edge_index = data.edge_index.numpy()
#     for i in range(edge_index.shape[1]):
#         G.add_edge(
#             edge_index[0, i],
#             edge_index[1, i],
#             length=edge_lengths[(edge_index[0, i], edge_index[1, i])],
#         )
#     return G


# # Visualize the initial graph and return positions
# def visualize_initial_graph(G, pos):
#     nx.draw(
#         G, pos, with_labels=True, node_size=500, node_color="blue"
#     )
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()


# # Create and show the initial graph
# G = create_nx_graph(data, edges)
# visualize_initial_graph(G, custom_pos)

# # Get user input for source and target
# source = int(input("Enter source node: "))
# target = int(input("Enter target node: "))

# # Create the model
# model = GCN(in_channels=4, hidden_channels=16, out_channels=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# # Training loop
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = F.nll_loss(out, data.y)
#     loss.backward()
#     optimizer.step()
#     return loss.item()


# # Training the model for 100 epochs
# for epoch in range(100):
#     loss = train()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss}")


# # Evaluation function
# def evaluate():
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         return pred


# # Get the predictions
# pred = evaluate()
# print(f"Predictions: {pred}")


# def find_optimal_path(G, source, target):
#     try:
#         path = nx.shortest_path(G, source=source, target=target, weight="length")
#     except nx.NetworkXNoPath:
#         path = [
#             source,
#             target,
#         ]  # If no path exists, return a direct connection (for visualization)

#     return path


# # Find the optimal path
# optimal_path = find_optimal_path(G, source, target)
# print(f"Optimal path from {source} to {target}: {optimal_path}")


# # Visualization function
# def visualize_graph(G, path, pos):
#     # Draw nodes
#     node_colors = ["red" if node in path else "blue" for node in G.nodes()]
#     nx.draw(
#         G,
#         pos,
#         node_color=node_colors,
#         with_labels=True,
#         node_size=500,
#         cmap=plt.cm.Blues,
#     )

#     # Highlight the path
#     path_edges = list(zip(path, path[1:]))
#     nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     plt.show()


# # Visualize the graph with the predicted path using the same layout
# visualize_graph(G, optimal_path, custom_pos)
