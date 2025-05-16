import sys, torch, os, json
sys.path.append("your/path")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import defaultdict


class GraphClassifier:
    def __init__(self, in_channels=128, hidden_channels=64, num_classes=4, distance=224):
        self.distance = distance
        self.model = SimpleGCN(in_channels, hidden_channels, num_classes)
        self.num_classes = num_classes
        self.device = torch.device("cpu")  

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self  

    def _is_neighbor(self, c1, c2):
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) == self.distance

    def _build_graph(self, coords, embeddings):
        num_nodes = len(coords)
        adj = defaultdict(list)
        sim_dict = {}

        norm_emb = F.normalize(embeddings, dim=1)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and self._is_neighbor(coords[i], coords[j]):
                    sim = torch.dot(norm_emb[i], norm_emb[j])
                    adj[i].append(j)
                    sim_dict[(i, j)] = sim.item()

        return adj, sim_dict

    def _build_edge_index(self, adj):
        edge_list = []
        for src, dsts in adj.items():
            for dst in dsts:
                edge_list.append((src, dst))
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def run_in_batches(self, data, batch_size=4, grad_enabled=True):
        all_logits = []

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            for node_list in batch:
                coords = [(d['location']['x'], d['location']['y']) for d in node_list]
                embeddings = torch.stack([
                    torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list
                ]).to(self.device)

                adj, sim_dict = self._build_graph(coords, embeddings)
                edge_index = self._build_edge_index(adj).to(self.device)

                if edge_index.size(1) == 0:
                    continue

                with torch.set_grad_enabled(grad_enabled):
                    out = self.model(embeddings, edge_index)  # [num_nodes, num_classes]
                    graph_logit = out.mean(dim=0)             # 图级表示
                    all_logits.append(graph_logit)

        if all_logits:
            return torch.stack(all_logits)
        else:
            return torch.empty((0, self.num_classes), device=self.device)


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.distance = 224

    def _is_neighbor(self, c1, c2):
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) == self.distance

    def _build_graph(self, coords, embeddings):
        num_nodes = len(coords)
        adj = defaultdict(list)
        sim_dict = {}

        norm_emb = F.normalize(embeddings, dim=1)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and self._is_neighbor(coords[i], coords[j]):
                    sim = torch.dot(norm_emb[i], norm_emb[j])
                    adj[i].append(j)
                    sim_dict[(i, j)] = sim.item()

        return adj, sim_dict

    def _build_edge_index(self, adj):
        edge_list = []
        for src, dsts in adj.items():
            for dst in dsts:
                edge_list.append((src, dst))
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x



if __name__ == "__main__":
    dir_path = f"data/embeddings/resnet"
    data_path = os.listdir(dir_path)[:10]

    data = []
    for path in data_path:
        file_path = os.path.join(dir_path, path)
        with open(file_path, "r") as f:
            sample = json.load(f)
            feature = [
                {
                    "location":d['location'],
                    "embedding":d['embedding']
                }
                for d in sample]
            label = [
                {
                    "site":d['site'],               # 10 classes
                    "subtype":d['subtype']          # 29 classes
                }
                for d in sample]
            data.append(sample)

    
    model = GraphClassifier(in_channels=2048, hidden_channels=512, num_classes=10) 

    logits = model.run_in_batches(data, batch_size=2)  # [num_graphs, num_classes]

    print("Logits shape:", logits.shape)
    print("Predictions:")
    preds = torch.argmax(logits, dim=1)
    print(preds)  