import torch
import math
from torch_scatter import scatter_add

class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Add self-loops to the edge index
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        # Compute the degree of each node
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()

        # Compute normalization coefficients
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Perform the linear transformation
        support = torch.mm(x, self.weight)

        # Efficient scatter add operation
        out = scatter_add(support[col] * norm.unsqueeze(1), row, dim=0, dim_size=num_nodes)

        # Add bias if present
        if self.bias is not None:
            out += self.bias
            
        return out

# Example usage
num_nodes = 1000  # Example number of nodes
num_edges = 5000  # Example number of edges
in_features = 16
out_features = 32
x = torch.rand((num_nodes, in_features), dtype=torch.float32)
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

layer = GraphConvolutionLayer(in_features, out_features)
output = layer(x, edge_index)

# Print output shape to verify
print("Output shape:", output.shape)
