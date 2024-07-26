import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.data import Data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.att1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
        self.att2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv2 = GCNConv(hidden_dim * 4, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.att1(x, edge_index)
        x = F.relu(x)
        x = self.att2(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_graph(num_nodes, edges):
    x = torch.eye(num_nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 9), 
    (1, 3), (2, 5), (3, 7), (4, 6), (5, 8), (6, 0), (7, 2), (8, 4), (9, 1)
]
num_nodes = 10
graph = create_graph(num_nodes, edges)
graphs = [graph]
dataloader = DataLoader(graphs, batch_size=1)

input_dim = num_nodes
hidden_dim = 64
output_dim = num_nodes
model = GNN(input_dim, hidden_dim, output_dim)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 500
source_node = 0
target_node = 9

# Custom loss function for path prediction
def path_loss(output, target, paths):
    loss = 0
    for path in paths:
        for i in range(len(path) - 1):
            current_output = output[path[i]].unsqueeze(0)
            target_index = torch.tensor([path[i + 1]], dtype=torch.long)
            loss += F.cross_entropy(current_output, target_index)
    return loss

# Example of known paths (for simplicity)
known_paths = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 3]]

for epoch in range(num_epochs):
    model.train()
    for data in dataloader:
        optimizer.zero_grad()
        
        output = model(data.x, data.edge_index, data.batch)
        
        # Compute custom loss based on known paths
        loss = path_loss(output, target_node, known_paths)
        
        loss.backward()
        optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Inference to find the most likely path
def find_path(output, source, target):
    path = [source]
    current = source
    while current != target:
        current = torch.argmax(output[current]).item()
        path.append(current)
    return path

model.eval()
for data in dataloader:
    output = model(data.x, data.edge_index, data.batch)
    predicted_path = find_path(output, source_node, target_node)
    print(f'Shortest path from node {source_node} to node {target_node} is predicted to be {predicted_path}')
