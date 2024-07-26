import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling

class FourierGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierGraphConvLayer, self).__init__()
        self.gcn1 = GCNConv(in_channels, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.gcn2 = GCNConv(64, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.gcn3 = GCNConv(128, out_channels)
    
    def forward(self, x, edge_index, batch):
        # First GCN layer with ReLU activation
        x = F.relu(self.gcn1(x, edge_index))
        # First pooling layer
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        
        # Second GCN layer with ReLU activation
        x = F.relu(self.gcn2(x, edge_index))
        # Second pooling layer
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        
        # Final GCN layer
        x = self.gcn3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Example usage:
# model = FourierGraphConvLayer(in_channels=dataset.num_features, out_channels=dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
