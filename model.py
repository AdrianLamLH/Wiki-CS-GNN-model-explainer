import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.datasets import WikiCS
embedding_size = 1024
graph = WikiCS(root=".")[0]

class GCN(torch.nn.Module):
    def __init__(in_graph):
        # Init parent
        super(GCN, in_graph).__init__()

        # GCN layers
        in_graph.dropout = nn.Dropout(p=0.5)
        in_graph.initial_conv = GCNConv(graph.num_features, embedding_size)
        in_graph.conv1 = GCNConv(embedding_size, embedding_size)

        # Output layer
        in_graph.fc = Linear(embedding_size, 10)
        in_graph.out = nn.Softmax()
        # in_graph.explainer_config = ('model',
        #     'attributes',
        #     'object')
        # in_graph.model_config = ('classification',
        #     'node',
        #     'log_probs'  # Model returns log probabilities.
        # )

    def forward(in_graph, x, edge_index):
        emb = in_graph.dropout(x)
        emb = F.relu(in_graph.initial_conv(emb, edge_index))
        emb = F.relu(in_graph.conv1(emb, edge_index))

        return in_graph.out(in_graph.fc(emb))
    
class GAT(torch.nn.Module):
    def __init__(in_graph):
        # Init parent
        super(GAT, in_graph).__init__()

        # GCN layers
        in_graph.initial_conv = GATConv(graph.num_features, embedding_size)
        in_graph.conv1 = GATConv(embedding_size, embedding_size)

        # Output layer
        in_graph.fc = Linear(embedding_size, 10)
        in_graph.out = torch.nn.Softmax()
        # in_graph.explainer_config = ('model',
        #     'attributes',
        #     'object')
        # in_graph.model_config = ('classification',
        #     'node',
        #     'log_probs'  # Model returns log probabilities.
        # )

    def forward(in_graph, x, edge_index):
        emb = F.relu(in_graph.initial_conv(x, edge_index))
        emb = F.relu(in_graph.conv1(emb, edge_index))

        return in_graph.out(in_graph.fc(emb))