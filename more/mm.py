import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, adjacent_matrix, feature_matrix):
        x = torch.mm(adjacent_matrix, feature_matrix)
        x = torch.relu(self.fc1(x))
        x = torch.mm(adjacent_matrix, x)
        x = self.fc2(x)
        return x


def optimal_path(adj_matrix, feature_matrix, source_node, target_node):
    num_nodes = feature_matrix.shape[0]
    input_dim = feature_matrix.shape[1]
    hidden_dim = 16
    output_dim = 1
    model = GNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(adj_matrix, feature_matrix)
        target = torch.zeros((num_nodes, output_dim))
        target[target_node] = 1
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        current_node = source_node
        path = [current_node]
        max_iter = num_nodes

        while current_node != target_node and max_iter > 0:
            neighbors = torch.nonzero(adj_matrix[current_node]).squeeze()
            if len(neighbors.shape) == 0 or neighbors.shape[0] == 0:
                break
            neigbor_scores = model(adj_matrix, feature_matrix)[neighbors]

            if len(neigbor_scores) == 0:
                break

            next_node = neighbors[torch.argmax(neigbor_scores)]
            current_node = next_node.item()
            path.append(current_node)
            max_iter -= 1

        return path, model.fc1.weight.detach().numpy()


def calculate_accuracy(path, target_path, adj_matrix):
    intersection = set(path).intersection(target_path)
    accuracy = len(intersection) / len(target_path)
    dijkstra_length = dijkstra_shortest_path(adj_matrix, source_node, target_node)
    return accuracy, dijkstra_length


def dijkstra_shortest_path(adj_matrix, source_node, target_node):
    graph = nx.from_numpy_array(adj_matrix)
    length = dijkstra_path_length(graph, source_node, target_node)
    return length


adjacency_matrix = np.array(
    [
        [0,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0]
    ],
    dtype=np.float32,
)

feature_matrix = np.array(
    [
        [218.75,70.104,1],
        [218.81,70.104,1.0423],
        [218.51,70.104,1.0423],
        [218.20,70.104,1.0423],
        [216.99,48.768,1.1896],
        [216.69,48.768,1.1896],
        [216.38,48.768,1.1896],
        [216.90,76.505,1.3312],
    ],
    dtype=np.float32,
)

source_node = 0
target_node = 7

adj_tensor = torch.from_numpy(adjacency_matrix)
feature_tensor = torch.from_numpy(feature_matrix)

target_path = [0, 1, 4, 7]
acc_values = []
dijkstra_len = []

try:
    for _ in range(10000):
        shortest_path,_ = optimal_path(adj_tensor, feature_tensor, source_node, target_node)
        acc, dijkstra_length = calculate_accuracy(shortest_path, target_path, adjacency_matrix)
        acc_values.append(acc)
        dijkstra_len.append(dijkstra_length)

    plt.figure(1)
    plt.plot(acc_values,label='GNN')
    plt.plot(dijkstra_len,label='Dijkstra')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/shortest_path')
    plt.title('GNN vs Dijkstra')
    plt.legend()
except Exception as e:
    print("yes")
    print(e)

graph_sizes = [10, 20, 30, 40, 50]
execution_times = []
dijkstra_execution_times = []

for size in graph_sizes:
    adjacency_matrix = np.ones((size, size),dtype=np.float32)
    feature_matrix = np.ones((size, 2),dtype=np.float32)
    adj_tensor = torch.from_numpy(adjacency_matrix)
    feature_tensor = torch.from_numpy(feature_matrix)

    start_time = time.time()

    shortest_path ,_ = optimal_path(adj_tensor, feature_tensor, source_node, target_node)
    execution_time = (time.time() - start_time )* 1000
    execution_times.append(execution_time)

    start_time = time.time()
    dijkstra_shortest_path(adjacency_matrix, source_node, target_node)
    dijkstra_execution_time = (time.time() - start_time) * 1000
    dijkstra_execution_times.append(dijkstra_execution_time)

plt.figure(3)
plt.plot(graph_sizes, execution_times, label='GNN')
plt.plot(graph_sizes, dijkstra_execution_times, label='Dijkstra')
plt.xlabel('Graph Size')
plt.ylabel('Execution Time (ms)')
plt.title('GNN vs Dijkstra scalability')
plt.legend()

_,learned_rep = optimal_path(adj_tensor, feature_tensor, source_node, target_node)
plt.figure(4)
plt.scatter(learned_rep[:,0],learned_rep[:,1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Learned Representation')
plt.show()
