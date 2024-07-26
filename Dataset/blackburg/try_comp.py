import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import heapq

# Define graph types and node lists
graph_types = ['random', 'grid', 'scale_free']
nodes_list = [10, 20, 30, 40, 50]  # Adjust based on your testing requirements

# Function for Dijkstra's algorithm
def dijkstra(G, source, target):
    if source not in G or target not in G:
        raise ValueError("Source or target node not in graph")
    
    queue, seen = [(0, source, [])], set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node not in seen:
            path = path + [node]
            seen.add(node)
            if node == target:
                return cost, path
            for next_node, data in G[node].items():
                if next_node not in seen:
                    heapq.heappush(queue, (cost + data['weight'], next_node, path))
    raise ValueError("Target node not reachable from source node")

# Function for A* algorithm
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star(G, source, target, pos):
    if source not in G or target not in G:
        raise ValueError("Source or target node not in graph")
    
    queue, seen = [(0, 0, source, [])], set()
    while queue:
        (cost, est, node, path) = heapq.heappop(queue)
        if node not in seen:
            path = path + [node]
            seen.add(node)
            if node == target:
                return cost, path
            for next_node, data in G[node].items():
                if next_node not in seen:
                    h = heuristic(pos[next_node], pos[target])
                    heapq.heappush(queue, (cost + data['weight'], cost + data['weight'] + h, next_node, path))
    raise ValueError("Target node not reachable from source node")

# Hypothetical GNN model function (for demonstration, ensuring valid path)
def gnn_model(G, source, target):
    nodes = list(G.nodes)
    if source not in nodes or target not in nodes:
        raise ValueError("Source or target node not in graph")
    
    path = [source]
    current = source
    while current != target:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            raise ValueError("No path found to target node")
        next_node = np.random.choice(neighbors)
        path.append(next_node)
        current = next_node
    cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    return cost, path

# Initialize edge weights
def initialize_edge_weights(G):
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)

# Measure scalability
def measure_scalability(nodes_list, graph_type):
    times_dijkstra, times_a_star, times_gnn = [], [], []
    for n in nodes_list:
        if graph_type == 'random':
            G = nx.random_geometric_graph(n, 0.3, seed=42)
            pos = nx.get_node_attributes(G, 'pos')
        elif graph_type == 'grid':
            G = nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n)))
            G = nx.convert_node_labels_to_integers(G)
            pos = {i: (i % int(np.sqrt(n)), i // int(np.sqrt(n))) for i in G.nodes()}
        elif graph_type == 'scale_free':
            G = nx.scale_free_graph(n)
            G = nx.convert_node_labels_to_integers(G)
            pos = nx.spring_layout(G, seed=42)
        
        # Initialize edge weights
        initialize_edge_weights(G)
        
        source, target = 0, n - 1
        
        try:
            start_time = time.time()
            dijkstra(G, source, target)
            times_dijkstra.append(time.time() - start_time)
        except ValueError as e:
            print(f"Dijkstra error for {graph_type} with {n} nodes: {e}")
            times_dijkstra.append(np.nan)
        
        try:
            start_time = time.time()
            a_star(G, source, target, pos)
            times_a_star.append(time.time() - start_time)
        except ValueError as e:
            print(f"A* error for {graph_type} with {n} nodes: {e}")
            times_a_star.append(np.nan)
        
        try:
            start_time = time.time()
            gnn_model(G, source, target)
            times_gnn.append(time.time() - start_time)
        except ValueError as e:
            print(f"GNN error for {graph_type} with {n} nodes: {e}")
            times_gnn.append(np.nan)
    
    return times_dijkstra, times_a_star, times_gnn

# Measure and plot scalability
plt.figure(figsize=(15, 5))

for graph_type in graph_types:
    times_dijkstra, times_a_star, times_gnn = measure_scalability(nodes_list, graph_type)
    plt.plot(nodes_list, times_dijkstra, label=f'Dijkstra ({graph_type})', marker='o')
    plt.plot(nodes_list, times_a_star, label=f'A* ({graph_type})', marker='o')
    plt.plot(nodes_list, times_gnn, label=f'GNN ({graph_type})', marker='o')

plt.xlabel('Number of Nodes')
plt.ylabel('Time (seconds)')
plt.title('Scalability Comparison')
plt.legend()
plt.grid(True)
plt.show()

# # Measure and plot adaptability
# plt.figure(figsize=(15, 5))

# for graph_type in graph_types:
#     times_dijkstra, times_a_star, times_gnn = measure_adaptability(graph_type)
#     plt.plot(range(10), times_dijkstra, label=f'Dijkstra ({graph_type})')
#     plt.plot(range(10), times_a_star, label=f'A* ({graph_type})')
#     plt.plot(range(10), times_gnn, label=f'GNN ({graph_type})')

# plt.xlabel('Test Case')
# plt.ylabel('Time (seconds)')
# plt.title('Adaptability Comparison')
# plt.legend()
# plt.show()

# # Measure and plot generalization
# unseen_graphs = [
#     nx.random_geometric_graph(25, 0.3, seed=43), 
#     nx.grid_2d_graph(6, 6), 
#     nx.scale_free_graph(30)
# ]
# graph_types_for_unseen = ['random', 'grid', 'scale_free']
# plt.figure(figsize=(15, 5))

# for unseen_graph, graph_type in zip(unseen_graphs, graph_types_for_unseen):
#     time_dijkstra, time_a_star, time_gnn = measure_generalization(graph_type, unseen_graph)
#     plt.bar([f'{graph_type}_Dijkstra'], [time_dijkstra], label='Dijkstra', color='red')
#     plt.bar([f'{graph_type}_A*'], [time_a_star], label='A*', color='blue')
#     plt.bar([f'{graph_type}_GNN'], [time_gnn], label='GNN', color='green')

# plt.ylabel('Time (seconds)')
# plt.title('Generalization Comparison')
# plt.legend()
# plt.show()


# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# import heapq

# # Generate a random graph
# G = nx.random_geometric_graph(25, 0.3, seed=42)
# pos = nx.get_node_attributes(G, 'pos')

# # Assign random weights to edges
# for (u, v, w) in G.edges(data=True):
#     w['weight'] = np.random.randint(1, 10)

# # Function for Dijkstra's algorithm
# def dijkstra(G, source, target):
#     queue, seen = [(0, source, [])], set()
#     while True:
#         (cost, node, path) = heapq.heappop(queue)
#         if node not in seen:
#             path = path + [node]
#             seen.add(node)
#             if node == target:
#                 return cost, path
#             for next_node, data in G[node].items():
#                 if next_node not in seen:
#                     heapq.heappush(queue, (cost + data['weight'], next_node, path))

# # Function for A* algorithm
# def heuristic(a, b):
#     (x1, y1) = a
#     (x2, y2) = b
#     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# def a_star(G, source, target):
#     queue, seen = [(0, 0, source, [])], set()
#     while True:
#         (cost, est, node, path) = heapq.heappop(queue)
#         if node not in seen:
#             path = path + [node]
#             seen.add(node)
#             if node == target:
#                 return cost, path
#             for next_node, data in G[node].items():
#                 if next_node not in seen:
#                     h = heuristic(pos[next_node], pos[target])
#                     heapq.heappush(queue, (cost + data['weight'], cost + data['weight'] + h, next_node, path))

# # Hypothetical GNN model function (for demonstration, ensuring valid path)
# def gnn_model(G, source, target):
#     nodes = list(G.nodes)
#     path = [source]
#     current = source
#     while current != target:
#         neighbors = list(G.neighbors(current))
#         next_node = np.random.choice(neighbors)
#         path.append(next_node)
#         current = next_node
#     cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
#     return cost, path

# # Compute paths
# source, target = 0, 24
# cost_dijkstra, path_dijkstra = dijkstra(G, source, target)
# cost_a_star, path_a_star = a_star(G, source, target)
# cost_gnn, path_gnn = gnn_model(G, source, target)

# # Plot the graph and paths
# plt.figure(figsize=(12, 8))

# nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=500, font_size=10, edge_color='gray')

# # Draw paths
# path_edges_dijkstra = list(zip(path_dijkstra, path_dijkstra[1:]))
# path_edges_a_star = list(zip(path_a_star, path_a_star[1:]))
# path_edges_gnn = list(zip(path_gnn, path_gnn[1:]))

# nx.draw_networkx_edges(G, pos, edgelist=path_edges_dijkstra, edge_color='red', width=2, label='Dijkstra')
# nx.draw_networkx_edges(G, pos, edgelist=path_edges_a_star, edge_color='blue', width=2, label='A*')
# nx.draw_networkx_edges(G, pos, edgelist=path_edges_gnn, edge_color='green', width=2, label='GNN')

# # Add legend
# plt.legend(['Dijkstra', 'A*', 'GNN'], loc='upper left')

# plt.title('Shortest Paths Comparison')
# plt.show()
