# ##########################################
#             # with loading weights ---
# ##########################################
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

# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.attn1(x, edge_index)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# # Define your custom graph data
# num_nodes = 12
# node_features = torch.tensor([
#     [218.75, 0.01341, 0.01768, 0.02211],
#     [218.45, 0.00913, 0.01341, 0.01768],
#     [218.15, 0.00485, 0.00913, 0.01355],
#     [217.81, 0.01671, 0.02282, 0.02914],
#     [217.51, 0.01060, 0.01671, 0.02303],
#     [217.20, 0.00428, 0.01040, 0.01671],
#     [216.99, 0.00118, 0.00523, 0.00915],
#     [216.69, 0, 0.00131, 0.00523],
#     [216.38, 0, 0, 0.00118],
#     [216.90, 0, 0, 0],
#     [216.59, 0, 0, 0],
#     [216.29, 0, 0, 0],
# ], dtype=torch.float)

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

# def create_random_graph(num_nodes, num_edges):
#     G = nx.gnm_random_graph(num_nodes, num_edges, seed=42)
#     pos = nx.spring_layout(G)
#     edge_lengths = {e: np.random.uniform(1, 10) for e in G.edges()}
#     nx.set_edge_attributes(G, edge_lengths, 'length')
#     return G, pos

# def visualize_initial_graph(G, pos):
#     nx.draw(G, pos, with_labels=True, node_size=500, node_color="blue")
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()

# def load_model():
#     model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
#     model.load_state_dict(torch.load('gcn_with_attention_model.pth'))
#     return model

# def evaluate_model(model, data):
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         return pred

# def find_optimal_path(G, source, target):
#     return nx.shortest_path(G, source=source, target=target, weight="length")

# def dijkstra_path(G, source, target):
#     return nx.dijkstra_path(G, source=source, target=target, weight='length')

# def astar_path(G, source, target):
#     def heuristic(u, v):
#         return distance.euclidean(custom_pos[u], custom_pos[v])
#     return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)

# def measure_performance(func, G, source, target):
#     start_time = time.time()
#     tracemalloc.start()
#     process = psutil.Process()

#     path = func(G, source, target)

#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     end_time = time.time()

#     computation_time = end_time - start_time
#     memory_usage = peak / 10**6  # Convert bytes to MB
#     cpu_usage = process.cpu_percent(interval=0.1)
#     nodes_explored = len(path)
    
#     return path, computation_time, nodes_explored, memory_usage, cpu_usage

# def plot_computation_time(times):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, times, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Computation Time (seconds)')
#     plt.title('Computation Time Comparison')
#     plt.show()

# def plot_nodes_explored(nodes_explored):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, nodes_explored, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Number of Nodes Explored')
#     plt.title('Nodes Explored Comparison')
#     plt.show()

# def plot_resource_usage(memory, cpu):
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     ax1.set_xlabel('Algorithm')
#     ax1.set_ylabel('Memory Usage (MB)', color='tab:blue')
#     ax1.bar(labels, memory, color='tab:blue', alpha=0.6, label='Memory Usage')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('CPU Usage (%)', color='tab:red')
#     ax2.plot(labels, cpu, color='tab:red', marker='o', label='CPU Usage')
#     ax2.tick_params(axis='y', labelcolor='tab:red')
#     fig.tight_layout()
#     plt.title('Resource Usage Comparison')
#     plt.show()

# def plot_scalability_comparison(num_nodes_list):
#     computation_times = []
#     memory_usages = []
#     for num_nodes in num_nodes_list:
#         G, _ = create_random_graph(num_nodes, num_nodes * 2)
#         source, target = 0, num_nodes - 1
#         path, dijkstra_time, nodes_explored, dijkstra_mem, dijkstra_cpu = measure_performance(dijkstra_path, G, source, target)
#         computation_times.append(dijkstra_time)
#         memory_usages.append(dijkstra_mem)
        
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(num_nodes_list, computation_times, marker='o', color='blue')
#     plt.xlabel('Number of Nodes')
#     plt.ylabel('Computation Time (seconds)')
#     plt.title('Scalability: Computation Time vs Number of Nodes')

#     plt.subplot(1, 2, 2)
#     plt.plot(num_nodes_list, memory_usages, marker='o', color='green')
#     plt.xlabel('Number of Nodes')
#     plt.ylabel('Memory Usage (MB)')
#     plt.title('Scalability: Memory Usage vs Number of Nodes')

#     plt.tight_layout()
#     plt.show()

# # Load model
# model = load_model()

# # Evaluate model
# pred = evaluate_model(model, data)
# print("Model Prediction:", pred)

# # Create and visualize graph
# G = create_nx_graph(data, edges)
# visualize_initial_graph(G, custom_pos)

# # Pathfinding
# source, target = 0, 11

# # Measure performance
# performance_results = {}
# algorithms = {
#     "Shortest Path": find_optimal_path,
#     "Dijkstra": dijkstra_path,
#     "A*": astar_path
# }

# for algo_name, algo_func in algorithms.items():
#     print(f"Testing {algo_name}...")
#     path, comp_time, nodes_explored, mem_usage, cpu_usage = measure_performance(algo_func, G, source, target)
#     performance_results[algo_name] = {
#         "Path": path,
#         "Computation Time": comp_time,
#         "Memory Usage (MB)": mem_usage,
#         "CPU Usage (%)": cpu_usage,
#         "Nodes Explored": nodes_explored
#     }

# # Print performance results
# for algo_name, results in performance_results.items():
#     print(f"\nAlgorithm: {algo_name}")
#     print(f"Optimal Path: {results['Path']}")
#     print(f"Computation Time: {results['Computation Time']} seconds")
#     print(f"Memory Usage: {results['Memory Usage (MB)']} MB")
#     print(f"CPU Usage: {results['CPU Usage (%)']}%")
#     print(f"Nodes Explored: {results['Nodes Explored']}")

# # Visualizations
# computation_times = [result['Computation Time'] for result in performance_results.values()]
# nodes_explored = [result['Nodes Explored'] for result in performance_results.values()]
# memory_usages = [result['Memory Usage (MB)'] for result in performance_results.values()]
# cpu_usages = [result['CPU Usage (%)'] for result in performance_results.values()]

# plot_computation_time(computation_times)
# plot_nodes_explored(nodes_explored)
# plot_resource_usage(memory_usages, cpu_usages)

# # Scalability comparison
# num_nodes_list = [10, 20, 30, 40, 50]
# plot_scalability_comparison(num_nodes_list)







###########################################
            #with loading weights 
###########################################

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

# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # GAT layers
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         # GCN layers
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Attention layers
#         x = self.attn1(x, edge_index)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index)

#         return F.log_softmax(x, dim=1)

# # Define your custom graph data
# num_nodes = 12
# num_edges = 27

# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     [
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

# def visualize_initial_graph(G, pos):
#     nx.draw(
#         G, pos, with_labels=True, node_size=500, node_color="blue"
#     )
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()

# # Create and show the initial graph
# G = create_nx_graph(data, edges)
# # visualize_initial_graph(G, custom_pos)

# # Get user input for source and target
# source = 0
# target = 11

# # Create the model
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
# # Load the trained model
# model.load_state_dict(torch.load('gcn_with_attention_model.pth'))
# model.eval()

# # Evaluation function
# def evaluate():
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
#         path = [source, target]  # If no path exists, return a direct connection (for visualization)
#     return path

# def dijkstra_path(G, source, target):
#     return nx.dijkstra_path(G, source=source, target=target, weight='length')

# def astar_path(G, source, target):
#     def heuristic(u, v):
#         return distance.euclidean(custom_pos[u], custom_pos[v])

#     return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)

# # Measure computation time and number of nodes explored
# def measure_performance(func, G, source, target):
#     start_time = time.time()
#     tracemalloc.start()
#     process = psutil.Process()

#     path = func(G, source, target)

#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     end_time = time.time()

#     computation_time = end_time - start_time
#     memory_usage = peak / 10**6  # Convert bytes to MB
#     cpu_usage = process.cpu_percent(interval=0.1)
#     nodes_explored = len(path)
    
#     return path, computation_time, memory_usage, cpu_usage, nodes_explored

# algorithms = {
#     "Shortest Path": find_optimal_path,
#     "Dijkstra": dijkstra_path,
#     "A*": astar_path
# }

# performance_results = {}
# for algo_name, algo_func in algorithms.items():
#     print(f"Testing {algo_name}...")
#     path, comp_time, mem_usage, cpu_usage, nodes_explored = measure_performance(algo_func, G, source, target)
#     performance_results[algo_name] = {
#         "Path": path,
#         "Computation Time": comp_time,
#         "Memory Usage (MB)": mem_usage,
#         "CPU Usage (%)": cpu_usage,
#         "Nodes Explored": nodes_explored
#     }

# # Print performance results
# for algo_name, results in performance_results.items():
#     print(f"\nAlgorithm: {algo_name}")
#     print(f"Optimal Path: {results['Path']}")
#     print(f"Computation Time: {results['Computation Time']} seconds")
#     print(f"Memory Usage: {results['Memory Usage (MB)']} MB")
#     print(f"CPU Usage: {results['CPU Usage (%)']}%")
#     print(f"Nodes Explored: {results['Nodes Explored']}")

# # Pie Chart of Proportions of Best Performance
# labels = performance_results.keys()
# sizes = [results["Computation Time"] for results in performance_results.values()]
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title("Proportion of Computation Time for Different Algorithms")
# plt.show()

# # Confusion Matrix
# true_labels = [1] * num_nodes  # Example true labels
# pred_labels = pred.numpy()
# conf_matrix = confusion_matrix(true_labels, pred_labels)
# disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()







##########################################
            # without loading weights 
##########################################
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

def visualize_initial_graph(G, pos):
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color="blue"
    )
    edge_labels = nx.get_edge_attributes(G, "length")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Create and show the initial graph
G = create_nx_graph(data, edges)
# visualize_initial_graph(G, custom_pos)

# Get user input for source and target
source = 0
# source = int(input("Enter source node: "))
target = 11

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
for epoch in range(1000):
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

def dijkstra_path(G, source, target):
    return nx.dijkstra_path(G, source=source, target=target, weight='length')

def astar_path(G, source, target):
    def heuristic(u, v):
        return distance.euclidean(custom_pos[u], custom_pos[v])

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





###########################################
            #without loading weights 
###########################################
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

# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # GAT layers
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         # GCN layers
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Attention layers
#         x = self.attn1(x, edge_index)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index)

#         return F.log_softmax(x, dim=1)

# # Define your custom graph data
# num_nodes = 12
# num_edges = 27

# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     [
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

# def visualize_initial_graph(G, pos):
#     nx.draw(
#         G, pos, with_labels=True, node_size=500, node_color="blue"
#     )
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()

# # Create and show the initial graph
# G = create_nx_graph(data, edges)
# # visualize_initial_graph(G, custom_pos)

# # Get user input for source and target
# source = 0
# # source = int(input("Enter source node: "))
# target = 11

# # Create the model
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
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
# for epoch in range(1000):
#     loss = train()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss}")

# # Save the model
# torch.save(model.state_dict(), 'gcn_with_attention_model.pth')

# # Load the model (for future use)
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
# model.load_state_dict(torch.load('gcn_with_attention_model.pth'))

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
#         path = [source, target]  # If no path exists, return a direct connection (for visualization)
#     return path

# def dijkstra_path(G, source, target):
#     return nx.dijkstra_path(G, source=source, target=target, weight='length')

# def astar_path(G, source, target):
#     def heuristic(u, v):
#         return distance.euclidean(custom_pos[u], custom_pos[v])

#     return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)

# # Measure computation time and number of nodes explored
# def measure_performance(func, G, source, target):
#     start_time = time.time()
#     tracemalloc.start()
#     process = psutil.Process()

#     path = func(G, source, target)

#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     end_time = time.time()

#     computation_time = end_time - start_time
#     memory_usage = peak / 10**6  # Convert bytes to MB
#     cpu_usage = process.cpu_percent(interval=0.1)
#     nodes_explored = len(path)
    
#     return path, computation_time, nodes_explored, memory_usage, cpu_usage

# # Find the optimal path from the model
# optimal_path, model_time, model_nodes, model_memory, model_cpu = measure_performance(find_optimal_path, G, source, target)
# print(f"Optimal path from {source} to {target} using model: {optimal_path}")

# # Find the path using Dijkstra's algorithm
# dijkstra_path_, dijkstra_time, dijkstra_nodes, dijkstra_memory, dijkstra_cpu = measure_performance(dijkstra_path, G, source, target)
# print(f"Dijkstra path from {source} to {target}: {dijkstra_path_}")

# # Find the path using A* algorithm
# astar_path_, astar_time, astar_nodes, astar_memory, astar_cpu = measure_performance(astar_path, G, source, target)
# print(f"A* path from {source} to {target}: {astar_path_}")

# # Visualize the paths
# def visualize_graph_comparison(G, paths, pos):
#     plt.figure(figsize=(12, 6))
#     nx.draw(G, pos, with_labels=True, node_size=500, node_color="blue")

#     colors = ['red', 'green', 'blue']
#     labels = ['Model', 'Dijkstra', 'A*']

#     # Draw edges and paths
#     for i, path in enumerate(paths):
#         edges_in_path = [(path[j], path[j+1]) for j in range(len(path)-1)]
#         nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color=colors[i], width=2.5, label=labels[i])

#     nx.draw_networkx_labels(G, pos)
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.legend()
#     plt.show()

# # Visualize the paths
# visualize_graph_comparison(G, [optimal_path, dijkstra_path_, astar_path_], custom_pos)

# # Path lengths
# dijkstra_length = sum(G[u][v]['length'] for u, v in zip(dijkstra_path_[:-1], dijkstra_path_[1:]))
# astar_length = sum(G[u][v]['length'] for u, v in zip(astar_path_[:-1], astar_path_[1:]))
# model_length = sum(G[u][v]['length'] for u, v in zip(optimal_path[:-1], optimal_path[1:]))

# # Comparison of path lengths
# def plot_path_lengths(lengths):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, lengths, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Path Length')
#     plt.title('Path Length Comparison')
#     plt.show()

# plot_path_lengths([model_length, dijkstra_length, astar_length])

# # Node-wise Path Coverage
# def node_coverage(paths):
#     coverage = {node: 0 for node in G.nodes()}
#     for path in paths:
#         for node in path:
#             coverage[node] += 1
#     return coverage

# paths = [optimal_path, dijkstra_path_, astar_path_]
# coverage = node_coverage(paths)

# def plot_node_coverage(coverage):
#     nodes = list(coverage.keys())
#     values = list(coverage.values())
    
#     plt.figure(figsize=(12, 6))
#     plt.bar(nodes, values, color='purple')
#     plt.xlabel('Node')
#     plt.ylabel('Number of Paths Covering Node')
#     plt.title('Node-wise Path Coverage')
#     plt.show()

# plot_node_coverage(coverage)

# # Heatmap of Path Weights
# def plot_heatmap(paths):
#     heatmap_data = np.zeros((num_nodes, num_nodes))
    
#     for path in paths:
#         for u, v in zip(path[:-1], path[1:]):
#             heatmap_data[u, v] = G[u][v]['length']
#             heatmap_data[v, u] = G[u][v]['length']  # since undirected graph

#     plt.figure(figsize=(12, 10))
#     plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Edge Weight')
#     plt.title('Heatmap of Path Weights')
#     plt.show()

# plot_heatmap([optimal_path, dijkstra_path_, astar_path_])

# # Computation Time Comparison
# def plot_computation_time(times):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, times, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Computation Time (seconds)')
#     plt.title('Computation Time Comparison')
#     plt.show()

# plot_computation_time([model_time, dijkstra_time, astar_time])

# # Nodes Explored Comparison
# def plot_nodes_explored(nodes_explored):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, nodes_explored, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Number of Nodes Explored')
#     plt.title('Nodes Explored Comparison')
#     plt.show()

# plot_nodes_explored([model_nodes, dijkstra_nodes, astar_nodes])

# # Resource Usage Comparison
# def plot_resource_usage(memory, cpu):
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']

#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     ax1.set_xlabel('Algorithm')
#     ax1.set_ylabel('Memory Usage (MB)', color='tab:blue')
#     ax1.bar(labels, memory, color='tab:blue', alpha=0.6, label='Memory Usage')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()
#     ax2.set_ylabel('CPU Usage (%)', color='tab:red')
#     ax2.plot(labels, cpu, color='tab:red', marker='o', label='CPU Usage')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     fig.tight_layout()
#     plt.title('Resource Usage Comparison')
#     plt.show()

# plot_resource_usage([model_memory, dijkstra_memory, astar_memory], [model_cpu, dijkstra_cpu, astar_cpu])

# # Accuracy vs Speed Comparison
# def plot_accuracy_vs_speed(lengths, times):
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']

#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     ax1.set_xlabel('Algorithm')
#     ax1.set_ylabel('Path Length (Accuracy)', color='tab:blue')
#     ax1.bar(labels, lengths, color='tab:blue', alpha=0.6, label='Path Length')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Computation Time (seconds)', color='tab:red')
#     ax2.plot(labels, times, color='tab:red', marker='o', label='Computation Time')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     fig.tight_layout()
#     plt.title('Accuracy vs Speed Comparison')
#     plt.show()

# plot_accuracy_vs_speed([model_length, dijkstra_length, astar_length], [model_time, dijkstra_time, astar_time])











###########################################
            #without loading weights 
###########################################
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.spatial import distance
# import time
# import numpy as np
# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # GAT layers
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         # GCN layers
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Attention layers
#         x = self.attn1(x, edge_index)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index)

#         return F.log_softmax(x, dim=1)

# # Define your custom graph data
# num_nodes = 12
# num_edges = 27

# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     [
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
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
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

# # Save the model
# torch.save(model.state_dict(), 'gcn_with_attention_model.pth')

# # Load the model (for future use)
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
# model.load_state_dict(torch.load('gcn_with_attention_model.pth'))

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
#         path = [source, target]  # If no path exists, return a direct connection (for visualization)
#     return path

# def dijkstra_path(G, source, target):
#     return nx.dijkstra_path(G, source=source, target=target, weight='length')

# def astar_path(G, source, target):
#     def heuristic(u, v):
#         return distance.euclidean(custom_pos[u], custom_pos[v])

#     return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)

# # Measure computation time and number of nodes explored
# def measure_performance(func, G, source, target):
#     start_time = time.time()
#     path = func(G, source, target)
#     end_time = time.time()
#     computation_time = end_time - start_time
#     nodes_explored = len(path)
#     return path, computation_time, nodes_explored

# # Find the optimal path from the model
# optimal_path, model_time, model_nodes = measure_performance(find_optimal_path, G, source, target)
# print(f"Optimal path from {source} to {target} using model: {optimal_path}")

# # Find paths using Dijkstra and A* algorithms
# dijkstra_path_, dijkstra_time, dijkstra_nodes = measure_performance(dijkstra_path, G, source, target)
# astar_path_, astar_time, astar_nodes = measure_performance(astar_path, G, source, target)

# print(f"Dijkstra path from {source} to {target}: {dijkstra_path_}")
# print(f"A* path from {source} to {target}: {astar_path_}")

# # Visualization function
# def visualize_graph_comparison(G, paths, pos):
#     plt.figure(figsize=(15, 10))

#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color="blue")

#     colors = ['red', 'green', 'blue']
#     labels = ['Model', 'Dijkstra', 'A*']

#     # Draw edges and paths
#     for i, path in enumerate(paths):
#         edges_in_path = [(path[j], path[j+1]) for j in range(len(path)-1)]
#         nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color=colors[i], width=2.5, label=labels[i])

#     nx.draw_networkx_labels(G, pos)
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.legend()
#     plt.show()

# # Visualize the paths
# visualize_graph_comparison(G, [optimal_path, dijkstra_path_, astar_path_], custom_pos)

# # Path lengths
# dijkstra_length = sum(G[u][v]['length'] for u, v in zip(dijkstra_path_[:-1], dijkstra_path_[1:]))
# astar_length = sum(G[u][v]['length'] for u, v in zip(astar_path_[:-1], astar_path_[1:]))
# model_length = sum(G[u][v]['length'] for u, v in zip(optimal_path[:-1], optimal_path[1:]))

# # Comparison of path lengths
# def plot_path_lengths(lengths):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, lengths, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Path Length')
#     plt.title('Path Length Comparison')
#     plt.show()

# plot_path_lengths([model_length, dijkstra_length, astar_length])

# # Node-wise Path Coverage
# def node_coverage(paths):
#     coverage = {node: 0 for node in G.nodes()}
#     for path in paths:
#         for node in path:
#             coverage[node] += 1
#     return coverage

# paths = [optimal_path, dijkstra_path_, astar_path_]
# coverage = node_coverage(paths)

# def plot_node_coverage(coverage):
#     nodes = list(coverage.keys())
#     values = list(coverage.values())
    
#     plt.figure(figsize=(12, 6))
#     plt.bar(nodes, values, color='purple')
#     plt.xlabel('Node')
#     plt.ylabel('Number of Paths Covering Node')
#     plt.title('Node-wise Path Coverage')
#     plt.show()

# plot_node_coverage(coverage)

# # Heatmap of Path Weights
# def plot_heatmap(paths):
#     heatmap_data = np.zeros((num_nodes, num_nodes))
    
#     for path in paths:
#         for u, v in zip(path[:-1], path[1:]):
#             heatmap_data[u, v] = G[u][v]['length']
#             heatmap_data[v, u] = G[u][v]['length']  # since undirected graph

#     plt.figure(figsize=(12, 10))
#     plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Edge Weight')
#     plt.title('Heatmap of Path Weights')
#     plt.show()

# plot_heatmap([optimal_path, dijkstra_path_, astar_path_])

# # Computation Time Comparison
# def plot_computation_time(times):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, times, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Computation Time (seconds)')
#     plt.title('Computation Time Comparison')
#     plt.show()

# plot_computation_time([model_time, dijkstra_time, astar_time])

# # Nodes Explored Comparison
# def plot_nodes_explored(nodes_explored):
#     plt.figure(figsize=(8, 5))
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     plt.bar(labels, nodes_explored, color=['blue', 'green', 'red'])
#     plt.xlabel('Algorithm')
#     plt.ylabel('Number of Nodes Explored')
#     plt.title('Nodes Explored Comparison')
#     plt.show()

# plot_nodes_explored([model_nodes, dijkstra_nodes, astar_nodes])








###########################################
            #without loading weights 
###########################################
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.spatial import distance


# class GCNWithAttention(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNWithAttention, self).__init__()
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # GAT layers
#         self.attn1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=True)
#         self.attn2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         # GCN layers
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Attention layers
#         x = self.attn1(x, edge_index)
#         x = F.relu(x)
#         x = self.attn2(x, edge_index)

#         return F.log_softmax(x, dim=1)


# # Define your custom graph data
# num_nodes = 12
# num_edges = 27

# # Node features: a matrix with shape [num_nodes, num_node_features]
# node_features = torch.tensor(
#     [
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


# def visualize_initial_graph(G, pos):
#     nx.draw(
#         G, pos, with_labels=True, node_size=500, node_color="blue"
#     )
#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()


# # Create and show the initial graph
# G = create_nx_graph(data, edges)
# # visualize_initial_graph(G, custom_pos)

# # Get user input for source and target
# source = 0
# target = 11
# # source = int(input("Enter source node: "))
# # target = int(input("Enter target node: "))

# # Create the model
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
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


# # Save the model
# torch.save(model.state_dict(), 'gcn_with_attention_model.pth')

# # Load the model (for future use)
# model = GCNWithAttention(in_channels=4, hidden_channels=16, out_channels=2)
# # model.load_state_dict(torch.load('gcn_with_attention_model.pth'))


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
#         path = [source, target]  # If no path exists, return a direct connection (for visualization)
#     return path


# def dijkstra_path(G, source, target):
#     return nx.dijkstra_path(G, source=source, target=target, weight='length')


# def astar_path(G, source, target):
#     def heuristic(u, v):
#         return distance.euclidean(custom_pos[u], custom_pos[v])

#     return nx.astar_path(G, source=source, target=target, weight='length', heuristic=heuristic)


# # Find the optimal path from the model
# optimal_path = find_optimal_path(G, source, target)
# print(f"Optimal path from {source} to {target} using model: {optimal_path}")

# # Find paths using Dijkstra and A* algorithms
# dijkstra_path_ = dijkstra_path(G, source, target)
# astar_path_ = astar_path(G, source, target)

# print(f"Dijkstra path from {source} to {target}: {dijkstra_path_}")
# print(f"A* path from {source} to {target}: {astar_path_}")


# # Visualization function
# def visualize_graph_comparison(G, paths, pos):
#     plt.figure(figsize=(15, 10))

#     # Draw nodes
#     nx.draw(
#         G,
#         pos,
#         node_size=500,
#         node_color='lightblue',
#         with_labels=True,
#     )

#     # Highlight all paths
#     colors = ['red', 'green', 'blue']
#     labels = ['GCN with Attention', 'Dijkstra', 'A*']
#     for path, color, label in zip(paths, colors, labels):
#         path_edges = list(zip(path, path[1:]))
#         nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2, label=label)

#     edge_labels = nx.get_edge_attributes(G, "length")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     plt.legend()
#     plt.title('Path Comparison')
#     plt.show()


# # Visualize the graph with all paths
# visualize_graph_comparison(G, [optimal_path, dijkstra_path_, astar_path_], custom_pos)
