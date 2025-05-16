import sys, torch, os, json
sys.path.append("your/path")
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx


class GraphWalker:
    def __init__(self, distance=224):
        self.distance = distance

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

    def _graph_walk(self, adj, sim_dict, num_nodes, threshold=0.1):
        visit_count = torch.zeros(num_nodes)
        walk_paths = []

        for start in range(num_nodes):
            current = start
            prev = -1
            path = [current]
            while True:
                visit_count[current] += 1
                neighbors = adj.get(current, [])
                if not neighbors:
                    break

                sim_neighbors = sorted(
                    [(n, sim_dict.get((current, n), -1)) for n in neighbors],
                    key=lambda x: x[1],
                    reverse=True
                )

                if not sim_neighbors:
                    break

                next_node, max_sim = sim_neighbors[0]

                if next_node == prev and len(sim_neighbors) > 1:
                    second_node, second_sim = sim_neighbors[1]
                    if second_sim + threshold < max_sim:
                        break
                    else:
                        next_node = second_node

                if next_node == prev or next_node in path:
                    break

                prev = current
                current = next_node
                path.append(current)

            walk_paths.append(path)

        return visit_count, walk_paths

    def analyze(self, coords, embeddings, threshold=0.1, visual=False):
        adj, sim_dict = self._build_graph(coords, embeddings)
        visit_count, walk_paths = self._graph_walk(adj, sim_dict, len(coords), threshold)

        if visual and coords:
            self.visualize_walk(coords, visit_count, walk_paths)

        weights = visit_count.unsqueeze(1)
        weighted_embeddings = embeddings * weights
        pooled = weighted_embeddings.sum(0) / weights.sum()
        return pooled, visit_count

    def visualize_walk(self, coords, visit_count, walk_paths):
        G = nx.DiGraph()

        for idx, (x, y) in enumerate(coords):
            G.add_node(idx, pos=(x, y))

        for path in walk_paths:
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1])

        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, edge_color='gray')

        sizes = (visit_count.numpy() - visit_count.min().item()) / (
            visit_count.max().item() - visit_count.min().item() + 1e-6
        )
        sizes = 100 + sizes * 800

        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=sizes, cmap='viridis', alpha=0.9)

        labels = {i: str(int(visit_count[i].item())) for i in range(len(coords))}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        plt.axis('off')
        plt.title("Graph Walk Visualization (Node Size ∝ Visit Count)")

        save_path = f"draw/walk_result.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()




class WSIGraphClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, pooled_embedding):
        return self.classifier(pooled_embedding)
    
    def run_in_batches(self, data, walker, batch_size=4, threshold=0.1, visual=False):
        all_logits = []

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            pools = []
            for j, node_list in enumerate(batch):
                coords = [(d['location']['x'], d['location']['y'])  for d in node_list]
                embeddings = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list])

                pooled, visit_count = walker.analyze(coords, embeddings, threshold=threshold, visual=visual)
                pools.append(pooled)

            batch_pool = torch.stack(pools, dim=0)
            logits = self.forward(batch_pool)
            all_logits.append(logits)

        return torch.concat(all_logits, dim=0)



if __name__ == "__main__":
    dir_path = f"data/embeddings/resnet"
    data_path = [os.listdir(dir_path)[43]]

    data = []
    for path in data_path:
        file_path = os.path.join(dir_path, path)
        print(file_path)
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
        print(label[0])

    walker = GraphWalker(distance=224)
    classifier = WSIGraphClassifier(embedding_dim=2048, num_classes=5)

    logits = classifier.run_in_batches(data=data, walker=walker, batch_size=1, threshold=0.1, visual=True)

    print(logits.shape)         # [B, num_classes]