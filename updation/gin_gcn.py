import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

class AdvancedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AdvancedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, batch=batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, batch=batch)
        x = self.conv3(x, edge_index)
        return x

# Create data
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 2, 3, 4]], dtype=torch.long)

batch = torch.zeros(x.size(0), dtype=torch.long)

data = Data(x=x, edge_index=edge_index, batch=batch)

# Create DataLoader
data_loader = DataLoader([data], batch_size=1)

# Model, optimizer, and loss
model = AdvancedGCN(in_channels=3, hidden_channels=64, out_channels=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    for batch_data in data_loader:
        out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        target = torch.tensor([0, 1, 0, 1, 0])  # Dummy targets
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

# Visualization functions
def visualize_graph(edge_index, node_features):
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    node_color = [node_features[i].tolist() for i in range(node_features.size(0))]
    nx.draw(G, pos, with_labels=True, node_color=node_color, cmap=plt.get_cmap('viridis'))
    nx.draw_networkx_edges(G, pos)
    plt.show()

def visualize_shortest_path(G, shortest_path):
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=[(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)], edge_color='r', width=2)
    plt.title("Shortest Path")
    plt.show()

# Visualize the graph
visualize_graph(edge_index, x)

# Create a NetworkX graph for path finding
G_nx = nx.Graph()
G_nx.add_edges_from(edge_index.t().tolist())

# Find shortest path
shortest_path = nx.shortest_path(G_nx, source=0, target=4)
visualize_shortest_path(G_nx, shortest_path)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GINConv, EdgeConv, TopKPooling, global_mean_pool

# class AdvancedGCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(AdvancedGCN, self).__init__()
#         self.gcn1 = GCNConv(in_channels, hidden_channels)
#         self.edge_conv1 = EdgeConv(nn=nn.Sequential(
#             nn.Linear(2 * hidden_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, hidden_channels)
#         ))
#         self.gin1 = GINConv(nn=nn.Sequential(
#             nn.Linear(hidden_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, hidden_channels)
#         ))
#         self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        
#         self.gcn2 = GCNConv(hidden_channels, hidden_channels)
#         self.edge_conv2 = EdgeConv(nn=nn.Sequential(
#             nn.Linear(2 * hidden_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, hidden_channels)
#         ))
#         self.gin2 = GINConv(nn=nn.Sequential(
#             nn.Linear(hidden_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, hidden_channels)
#         ))
#         self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        
#         self.fc = nn.Linear(hidden_channels, out_channels)
    
#     def forward(self, x, edge_index, batch):
#         # Initial Graph Convolution
#         x = F.relu(self.gcn1(x, edge_index))
        
#         # Edge Convolution
#         x = F.relu(self.edge_conv1(x, edge_index))
        
#         # Graph Isomorphism Network
#         x = F.relu(self.gin1(x, edge_index))
        
#         # First pooling layer
#         x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        
#         # Second round of GCN, EdgeConv, GIN
#         x = F.relu(self.gcn2(x, edge_index))
#         x = F.relu(self.edge_conv2(x, edge_index))
#         x = F.relu(self.gin2(x, edge_index))
        
#         # Second pooling layer
#         x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        
#         # Global mean pooling
#         x = global_mean_pool(x, batch)
        
#         # Final fully connected layer
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# Example usage:
# model = AdvancedGCN(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
