import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt  # Add matplotlib for plotting
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define the Graph Neural Network
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values, dim=1).item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            q_values = self.q_network(state)
            next_q_values = self.target_network(next_state)
            q_value = q_values[0][action]

            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * torch.max(next_q_values)

            loss = self.loss_fn(q_value, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Environment setup
def create_graph():
    G = nx.grid_2d_graph(5, 5)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G

def get_state(graph, node):
    state = np.zeros(len(graph.nodes))
    state[node] = 1
    return state, node  # Modified to return state and node ID

def main():
    graph = create_graph()
    state_dim = len(graph.nodes)
    action_dim = len(graph.nodes)
    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 500
    batch_size = 32

    paths = []  # Track paths chosen by the agent

    for episode in range(num_episodes):
        node = random.choice(list(graph.nodes))
        state, start_node = get_state(graph, node)  # Get initial state and starting node
        total_reward = 0
        path = [start_node]  # Track the path for visualization

        for t in range(200):
            action = agent.act(state)
            next_node = action if graph.has_edge(node, action) else node
            reward = 1 if next_node == action else -1
            next_state, next_node_id = get_state(graph, next_node)
            done = next_node == action

            agent.remember(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            node = next_node
            total_reward += reward
            path.append(next_node_id)  # Append the node ID to the path

            if done:
                break

        agent.update_target_network()
        paths.append(path)  # Store the path for this episode
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Plotting the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)  # Layout for the graph visualization
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='grey', linewidths=1, font_size=10)

    # Plotting the paths chosen
    for path in paths:
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='b', width=2)

    plt.title('Paths Chosen by the Agent on the Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
