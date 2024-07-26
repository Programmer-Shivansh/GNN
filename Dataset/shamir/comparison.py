
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import psutil
import tracemalloc
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from graph import nodes ,elevation ,edges,pos

node_list =[]
for node in nodes:
    node_list.append(list(nodes[node]))

if len(elevation) == len(node_list):
    for i in range(len(node_list)):
        node_list[i].append(elevation[i])

egde_dict ={}      
for i in edges:
    egde_dict[(i[0],i[1])] = i[2]
    # egde_dict[(i[0],i[1])] = [i[2],i[3]]


class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNWithAttention, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # GAT layers
        self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
        self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index_model):
        # GCN layers

        x = self.conv1(x, edge_index_model)
        # print("########################")
        x = F.relu(x)
        x = self.conv2(x, edge_index_model)
        x = F.relu(x)

        # Attention layers
        x = self.attn1(x, edge_index_model)
        x = F.relu(x)
        x = self.attn2(x, edge_index_model)

        return F.log_softmax(x, dim=1)

# Define your custom graph data
# num_nodes = 12
# num_edges = 27
num_nodes = len(node_list)
num_edges = len(egde_dict)
# Node features: a matrix with shape [num_nodes, num_node_features]
node_features = torch.tensor(
    node_list, dtype=torch.float
)

# Edge indices and custom lengths
edges =egde_dict
edge_index_my = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()

# Labels for the nodes (e.g., part of the optimal path or not)
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

def visualize_initial_graph(G, pos):
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color="blue"
    )
    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Create and show the initial graph
G = create_nx_graph(data, edges)
# visualize_initial_graph(G, pos)

# Get user input for source and target
source = 1
# source = int(input("Enter source node: "))
target = 6

# Create the model
model = GCNWithAttention(in_channels=3, hidden_channels=30, out_channels=2)
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

def dijkstra_path(G, source, target):
    return nx.dijkstra_path(G, source=source, target=target, weight='length')

def astar_path(G, source, target):
    def heuristic(u, v):
        return distance.euclidean(pos[u], pos[v])

    return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)

# Measure computation time and number of nodes explored
def measure_performance(func, G, source, target):
    start_time = time.time()
    tracemalloc.start()
    process = psutil.Process()

    path = func(G, source, target)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    computation_time = end_time - start_time
    memory_usage = peak / 10**6  # Convert bytes to MB
    cpu_usage = process.cpu_percent(interval=0.1)
    nodes_explored = len(path)
    
    return path, computation_time, memory_usage, cpu_usage, nodes_explored

algorithms = {
    "Shortest Path": find_optimal_path,
    "Dijkstra": dijkstra_path,
    "A*": astar_path
}

performance_results = {}
for algo_name, algo_func in algorithms.items():
    print(f"Testing {algo_name}...")
    path, comp_time, mem_usage, cpu_usage, nodes_explored = measure_performance(algo_func, G, source, target)
    performance_results[algo_name] = {
        "Path": path,
        "Computation Time": comp_time,
        "Memory Usage (MB)": mem_usage,
        "CPU Usage (%)": cpu_usage,
        "Nodes Explored": nodes_explored
    }

# Print performance results
for algo_name, results in performance_results.items():
    print(f"\nAlgorithm: {algo_name}")
    print(f"Optimal Path: {results['Path']}")
    print(f"Computation Time: {results['Computation Time']} seconds")
    print(f"Memory Usage: {results['Memory Usage (MB)']} MB")
    print(f"CPU Usage: {results['CPU Usage (%)']}%")
    print(f"Nodes Explored: {results['Nodes Explored']}")

# Pie Chart of Proportions of Best Performance
labels = performance_results.keys()
sizes = [results["Computation Time"] for results in performance_results.values()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Proportion of Computation Time for Different Algorithms")
plt.show()

# Confusion Matrix
true_labels = [1] * num_nodes  # Example true labels
pred_labels = pred.numpy()
conf_matrix = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

def test_scalability(graph_sizes):
    scalability_results = []
    for size in graph_sizes:
        # Generate a random graph with the given size
        G = nx.erdos_renyi_graph(size, 0.5)
        edge_lengths = {edge: np.random.random() for edge in G.edges}
        pos = nx.spring_layout(G)  # Use a layout for visualization
        
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        node_features = torch.rand(size, 3)  # Random features for the nodes
        data = Data(x=node_features, edge_index=edge_index)
        
        # Measure performance
        path, comp_time, mem_usage, cpu_usage, nodes_explored = measure_performance(find_optimal_path, G, source, target)
        scalability_results.append((size, comp_time, mem_usage, cpu_usage))
    
    return scalability_results

# Test scalability for different graph sizes
graph_sizes = [50, 100, 200, 400, 800]  # Example sizes
scalability_results = test_scalability(graph_sizes)

# Plot scalability results
sizes, times, mems, cpus = zip(*scalability_results)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(sizes, times, marker='o')
plt.xlabel('Graph Size')
plt.ylabel('Computation Time (s)')
plt.title('Scalability - Computation Time')

plt.subplot(1, 3, 2)
plt.plot(sizes, mems, marker='o')
plt.xlabel('Graph Size')
plt.ylabel('Memory Usage (MB)')
plt.title('Scalability - Memory Usage')

plt.subplot(1, 3, 3)
plt.plot(sizes, cpus, marker='o')
plt.xlabel('Graph Size')
plt.ylabel('CPU Usage (%)')
plt.title('Scalability - CPU Usage')

plt.tight_layout()
plt.show()

def test_adaptability(densities):
    adaptability_results = []
    for density in densities:
        # Generate a random graph with a given density
        size = 100  # Fixed size for simplicity
        G = nx.erdos_renyi_graph(size, density)
        edge_lengths = {edge: np.random.random() for edge in G.edges}
        pos = nx.spring_layout(G)  # Use a layout for visualization
        
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        node_features = torch.rand(size, 3)  # Random features for the nodes
        data = Data(x=node_features, edge_index=edge_index)
        
        # Measure performance
        path, comp_time, mem_usage, cpu_usage, nodes_explored = measure_performance(find_optimal_path, G, source, target)
        adaptability_results.append((density, comp_time, mem_usage, cpu_usage))
    
    return adaptability_results

# Test adaptability for different edge densities
densities = [0.1, 0.2, 0.4, 0.6, 0.8]  # Example densities
adaptability_results = test_adaptability(densities)

# Plot adaptability results
dens, times, mems, cpus = zip(*adaptability_results)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(dens, times, marker='o')
plt.xlabel('Edge Density')
plt.ylabel('Computation Time (s)')
plt.title('Adaptability - Computation Time')

plt.subplot(1, 3, 2)
plt.plot(dens, mems, marker='o')
plt.xlabel('Edge Density')
plt.ylabel('Memory Usage (MB)')
plt.title('Adaptability - Memory Usage')

plt.subplot(1, 3, 3)
plt.plot(dens, cpus, marker='o')
plt.xlabel('Edge Density')
plt.ylabel('CPU Usage (%)')
plt.title('Adaptability - CPU Usage')

plt.tight_layout()
plt.show()

def test_generalization(num_graphs):
    generalization_results = []
    for _ in range(num_graphs):
        size = 100  # Fixed size for simplicity
        G = nx.erdos_renyi_graph(size, 0.5)
        edge_lengths = {edge: np.random.random() for edge in G.edges}
        pos = nx.spring_layout(G)  # Use a layout for visualization
        
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        node_features = torch.rand(size, 3)  # Random features for the nodes
        data = Data(x=node_features, edge_index=edge_index)
        
        # Measure performance
        path, comp_time, mem_usage, cpu_usage, nodes_explored = measure_performance(find_optimal_path, G, source, target)
        generalization_results.append((comp_time, mem_usage, cpu_usage))
    
    return generalization_results

# Test generalization across multiple random graphs
num_graphs = 10  # Number of graphs to test
generalization_results = test_generalization(num_graphs)

# Plot generalization results
times, mems, cpus = zip(*generalization_results)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.hist(times, bins=10)
plt.xlabel('Computation Time (s)')
plt.ylabel('Frequency')
plt.title('Generalization - Computation Time')

plt.subplot(1, 3, 2)
plt.hist(mems, bins=10)
plt.xlabel('Memory Usage (MB)')
plt.ylabel('Frequency')
plt.title('Generalization - Memory Usage')

plt.subplot(1, 3, 3)
plt.hist(cpus, bins=10)
plt.xlabel('CPU Usage (%)')
plt.ylabel('Frequency')
plt.title('Generalization - CPU Usage')

plt.tight_layout()
plt.show()
