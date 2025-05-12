import sys, torch, os, json
sys.path.append("your/path")
import torch
import torch.nn as nn
import torch.nn.functional as F




class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        # Patch embedding层（可学习的变换）
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Attention层：用于为每个 patch 分配权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类器：对加权聚合后的图表示进行分类
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cpu")  # 默认值

    def forward(self, x):
        """
        x: Tensor of shape [num_patches, input_dim]
        """
        h = self.embedding(x)            # shape: [num_patches, hidden_dim]
        a = self.attention(h)            # shape: [num_patches, 1]
        a = torch.softmax(a, dim=0)      # 归一化 attention 权重
        z = torch.sum(a * h, dim=0)      # 加权求和，shape: [hidden_dim]
        out = self.classifier(z)         # shape: [num_classes]
        return out  # 每张图的预测结果
    
    def run_in_batches(self, data, batch_size=4, grad_enabled=True):
        """
        data: List of List[dict]，每个样本是一个 patch 组成的 list，每个 patch 是一个 dict，含有 'embedding'
        batch_size: 每次处理几个 WSI 图
        device: 推理设备（默认当前模型设备）
        """
        all_logits = []

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            for node_list in batch:
                # 构建该图的 patch embedding 输入张量
                embeddings = torch.stack([
                    torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list
                ]).to(self.device)

                if embeddings.size(0) == 0:
                    continue  # 跳过空图

                with torch.set_grad_enabled(grad_enabled):
                    logit = self.forward(embeddings)  # 每张图得到一个 [num_classes] 的 logit
                    all_logits.append(logit)

        if all_logits:
            return torch.stack(all_logits)  # shape: [num_graphs, num_classes]
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
            data.append(torch.stack(patch_list))  # [num_patches, embed_dim]

    # 初始化模型
    model = ABMIL(input_dim=2048, hidden_dim=512, num_classes=10)  # 适用于 site 分类
    model.eval()

    all_logits = []
    with torch.no_grad():
        for patch_embed in data:
            logit = model(patch_embed)  # shape: [num_classes]
            all_logits.append(logit)

    logits = torch.stack(all_logits)  # shape: [num_graphs, num_classes]
    preds = torch.argmax(logits, dim=1)
    print("Predictions:", preds)