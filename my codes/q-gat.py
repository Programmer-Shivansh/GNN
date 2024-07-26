import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import networkx as nx
from node2vec import Node2Vec

# Define the Graph Attention Network with attention mechanism
class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize node embeddings using Node2Vec
def initialize_node_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[str(node)] for node in G.nodes()}
    return embeddings

# Create a graph using NetworkX and initialize node features
def create_graph():
    G = nx.grid_2d_graph(5, 5)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    embeddings = initialize_node_embeddings(G)
    for node, embedding in embeddings.items():
        G.nodes[node]['feature'] = torch.tensor(embedding, dtype=torch.float)
    return G

# Convert NetworkX graph to PyTorch Geometric data
def graph_to_data(G):
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    data.x = torch.stack([G.nodes[node]['feature'] for node in G.nodes()])
    return data

# Define the DQN Agent with a GNN
class DQNAgent:
    def __init__(self, state_dim, action_dim, model, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = model
        self.target_network = model
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, graph_data):
        if np.random.rand() <= self.epsilon:
            return torch.tensor([random.randrange(self.action_dim)])  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(graph_data)
        return torch.argmax(q_values, dim=1)  # Choose action with maximum Q-value

    def train(self, batch_size, graph_data):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            q_values = self.q_network(graph_data)
            next_q_values = self.target_network(graph_data)
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
def main():
    graph = create_graph()
    data = graph_to_data(graph)
    state_dim = data.num_nodes
    action_dim = data.num_nodes

    model = GraphAttentionNetwork(input_dim=64, hidden_dim=128, output_dim=action_dim, heads=4)
    agent = DQNAgent(state_dim, action_dim, model)

    num_episodes = 500
    batch_size = 32

    for episode in range(num_episodes):
        node = random.choice(list(graph.nodes))
        state = data.x[node].numpy()
        total_reward = 0

        for t in range(200):
            action = agent.act(state, data)
            action_index = action.item()  # Convert tensor to scalar index
            next_node = action_index if graph.has_edge(node, action_index) else node
            reward = 1 if next_node == action_index else -1
            next_state = data.x[next_node].numpy()
            done = next_node == action_index

            agent.remember(state, action_index, reward, next_state, done)
            agent.train(batch_size, data)

            state = next_state
            node = next_node
            total_reward += reward

            if done:
                break

        agent.update_target_network()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
