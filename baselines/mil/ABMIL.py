import sys, torch, os, json
sys.path.append("your/path")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cpu")

    def forward(self, x):
        h = self.embedding(x)
        a = self.attention(h)
        a = torch.softmax(a, dim=0)
        z = torch.sum(a * h, dim=0)
        out = self.classifier(z)
        return out

    def run_in_batches(self, data, batch_size=4, grad_enabled=True):
        all_logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            for node_list in batch:
                embeddings = torch.stack([
                    torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list
                ]).to(self.device)
                if embeddings.size(0) == 0:
                    continue
                with torch.set_grad_enabled(grad_enabled):
                    logit = self.forward(embeddings)
                    all_logits.append(logit)
        if all_logits:
            return torch.stack(all_logits)
        else:
            return torch.empty((0, self.classifier.out_features), device=self.device)

if __name__ == "__main__":
    dir_path = "a5037_course_work/data/embeddings/resnet"
    data_path = os.listdir(dir_path)[:10]
    data = []
    for path in data_path:
        file_path = os.path.join(dir_path, path)
        with open(file_path, "r") as f:
            sample = json.load(f)
            patch_list = [
                torch.tensor(d["embedding"], dtype=torch.float32)
                for d in sample
            ]
            data.append(torch.stack(patch_list))
    model = ABMIL(input_dim=2048, hidden_dim=512, num_classes=10)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for patch_embed in data:
            logit = model(patch_embed)
            all_logits.append(logit)
    logits = torch.stack(all_logits)
    preds = torch.argmax(logits, dim=1)
    print("Predictions:", preds)