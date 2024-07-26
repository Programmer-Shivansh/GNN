import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv, TopKPooling, global_mean_pool

class AdvancedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AdvancedGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.edge_conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_channels)
        ))
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_channels)
        ))
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        # Initial Graph Convolution
        x = F.relu(self.gcn1(x, edge_index))
        
        # Edge Convolution
        x = F.relu(self.edge_conv1(x, edge_index))
        
        # First pooling layer
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        
        # Second round of GCN and EdgeConv
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.edge_conv2(x, edge_index))
        
        # Second pooling layer
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Final fully connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage:
# model = AdvancedGCN(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
