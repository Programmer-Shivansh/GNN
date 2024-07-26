import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

# Normalizing coordinates
nodes = {
    1: (4234.93, 4513.14), 2: (6939.72, 7557.96), 3: (8531.68, 9644.51),
    4: (200.93, 3508.50), 5: (1267.39, 4497.68), 6: (1916.54, 5146.83),
    7: (2720.25, 5780.53), 8: (3400.31, 5224.11), 9: (4806.80, 8114.37),
    10: (5363.21, 7635.24), 11: (5919.63, 8315.30), 12: (5301.39, 8639.88),
    13: (6460.59, 6429.68), 14: (6429.68, 4559.51), 15: (6707.88, 9211.75),
    16: (5734.16, 9907.26), 17: (7573.42, 10154.56), 18: (9026.28, 6491.50),
    19: (10540.96, 6491.50), 20: (2812.98, 4636.79), 21: (2009.27, 3848.53),
    22: (7619.78, 6429.68), 23: (7697.06, 4574.96), 24: (9814.53, 4930.45),
    25: (4095.83, 8825.35), 26: (4173.11, 7387.94), 27: (4729.52, 6939.72),
    28: (3384.85, 8129.83), 29: (3400.31, 6615.15), 30: (4018.55, 6151.47)
}

coords = np.array(list(nodes.values()))
x = coords[:, 0]
y = coords[:, 1]

x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

node_features = torch.tensor(np.column_stack((x_normalized, y_normalized)), dtype=torch.float)

# Define edges with attributes
edges = [
    (1, 4, 415.44, 203.20), (1, 8, 46.02, 101.60), (2, 3, 270.66, 152.40), 
    (2, 11, 47.24, 101.60), (2, 13, 94.18, 101.60), (3, 17, 213.06, 25.40), 
    (3, 18, 350.83, 152.40), (4, 5, 334.67, 152.40), (5, 6, 176.17, 152.40),
    (5, 21, 186.23, 152.40), (6, 7, 28.96, 152.40), (7, 8, 127.71, 76.20),
    (7, 29, 63.40, 152.40), (8, 30, 33.53, 50.80), (9, 10, 137.47, 152.40),
    (9, 12, 17.98, 50.80), (9, 25, 126.80, 152.40), (11, 15, 92.35, 304.80),
    (12, 11, 250.85, 25.40), (13, 14, 233.48, 203.20), (14, 23, 116.43, 25.40),
    (15, 16, 231.04, 254.00), (15, 17, 307.54, 25.40), (18, 19, 124.36, 152.40),
    (18, 24, 359.97, 25.40), (21, 20, 34.44, 25.40), (22, 13, 213.67, 152.40),
    (23, 22, 106.99, 152.40), (24, 19, 294.74, 152.40), (26, 9, 82.60, 76.20),
    (26, 27, 96.62, 152.40), (26, 28, 129.24, 152.40), (29, 26, 222.50, 152.40)
]

edge_index = torch.tensor([(e[0] - 1, e[1] - 1) for e in edges], dtype=torch.long).t().contiguous()

# Labels for the nodes (e.g., part of the optimal path or not)
y = torch.zeros(len(nodes), dtype=torch.long)

data = Data(x=node_features, edge_index=edge_index, y=y)

# Adjust node indices to be 0-based in custom_pos
custom_pos = {i-1: (x_normalized[i-1], y_normalized[i-1]) for i in nodes.keys()}

# Visualization function
def visualize_initial_graph(G, pos):
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color="blue"
    )
    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Create and show the initial graph


def create_nx_graph(data, edges):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(
            edge_index[0, i],
            edge_index[1, i],
            length=edges[i][2],
            diameter=edges[i][3]
        )
    return G

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
source = 0
# source = int(input("Enter source node: "))
target = 29

# Create the model
model = GCNWithAttention(in_channels=2, hidden_channels=16, out_channels=2)
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
for epoch in range(1000):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Save the model
# torch.save(model.state_dict(), 'gcn_with_attention_model.pth')

# Load the model (for future use)
model = GCNWithAttention(in_channels=2, hidden_channels=16, out_channels=2)
# model.load_state_dict(torch.load('gcn_with_attention_model.pth'))
