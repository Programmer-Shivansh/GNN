import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph import nodes, elevation, edges, pos

# Prepare node features and edges
node_list = [list(nodes[node]) + [elevation[i]] for i, node in enumerate(nodes)]
edge_dict = {(i[0], i[1]): i[2] for i in edges}

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNWithAttention, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
        self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index_model):
        src, dest = edge_index_model
        elevation_src = x[src, 2]
        elevation_dest = x[dest, 2]
        valid_edges = elevation_dest < elevation_src
        edge_index_filtered = edge_index_model[:, valid_edges]
        x = self.conv1(x, edge_index_filtered)
        x = F.relu(x)
        x = self.conv2(x, edge_index_filtered)
        x = F.relu(x)
        x = self.attn1(x, edge_index_filtered)
        x = F.relu(x)
        x = self.attn2(x, edge_index_filtered)
        return F.log_softmax(x, dim=1), edge_index_filtered

# Define graph data
num_nodes = len(node_list)
node_features = torch.tensor(node_list, dtype=torch.float)
edge_index_my = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
y = torch.zeros(num_nodes, dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index_my, y=y)

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

def visualize_nodes(G, pos, highlighted_nodes, start, end):
    node_colors = [
        'red' if node in highlighted_nodes else
        'green' if node == start else
        'blue' if node == end else
        'lightblue'
        for node in G.nodes()
    ]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors)
    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Create and show the initial graph
G = create_nx_graph(data, edge_dict)

# Get user input for source and target
source = 23
target = 2

# Create the model
model = GCNWithAttention(in_channels=3, hidden_channels=30, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training the model
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

def evaluate():
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Determine which nodes are part of the predicted path
        highlighted_nodes = set(np.where(pred == 1)[0])
        
        # Ensure the start and end nodes are highlighted
        return pred, highlighted_nodes

# Get the predictions and highlighted nodes
pred, highlighted_nodes = evaluate()

# Always highlight the start and end nodes
highlighted_nodes.update([source, target])

# Visualize the graph with the highlighted nodes
visualize_nodes(G, pos, highlighted_nodes, source, target)

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.spatial import distance
# import time
# import psutil
# import tracemalloc
# import numpy as np
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from graph import nodes ,elevation ,edges,pos

# node_list =[]
# for node in nodes:
#     node_list.append(list(nodes[node]))

# if len(elevation) == len(node_list):
#     for i in range(len(node_list)):
#         node_list[i].append(elevation[i])

# egde_dict ={}      
# for i in edges:
#     egde_dict[(i[0],i[1])] = i[2]
#     # egde_dict[(i[0],i[1])] = [i[2],i[3]]


# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # GAT layers
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index_model):
#         # Filter edges based on elevation constraint
#         src, dest = edge_index_model
#         elevation_src = x[src, 2]
#         elevation_dest = x[dest, 2]
#         valid_edges = elevation_dest < elevation_src

#         edge_index_filtered = edge_index_model[:, valid_edges]

#         # GCN layers
#         x = self.conv1(x, edge_index_filtered)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index_filtered)
#         x = F.relu(x)

#         # Attention layers
#         x = self.attn1(x, edge_index_filtered)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index_filtered)

#         return F.log_softmax(x, dim=1)

# # Define your custom graph data
# num_nodes = len(node_list)
# num_edges = len(egde_dict)
# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     node_list, dtype=torch.float
# )

# # Edge indices and custom lengths
# edges =egde_dict
# edge_index_my = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()

# # Labels for the nodes (e.g., part of the optimal path or not)
# y = torch.zeros(num_nodes, dtype=torch.long)
# data = Data(x=node_features, edge_index=edge_index_my, y=y)
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

# def visualize_initial_graph(G, pos):
#     nx.draw(
#         G, pos, with_labels=True, node_size=500, node_color="blue"
#     )
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()

# # Create and show the initial graph
# G = create_nx_graph(data, edges)
# # visualize_initial_graph(G, pos)

# # Get user input for source and target
# source = 1
# # source = int(input("Enter source node: "))
# target = 29

# # Create the model
# model = GCNWithAttention(in_channels=3, hidden_channels=30, out_channels=2)
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

# def evaluate():
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         return pred

# # Get the predictions
# pred = evaluate()
# print(f"Predictions: {pred}")
