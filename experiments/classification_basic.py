import torch, argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os
import sys
sys.path.append("your/path")
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from a5037_course_work.src.models.WSI_graph_walk import WSIGraphClassifier


site_subtype = [
    ("brain", "GBM"), ("brain", "LGG"), 
    ("gastrointestinal", "COAD"), ("gastrointestinal", "READ"), ("gastrointestinal", "ESCA"), ("gastrointestinal", "STAD"), 
    ("gynecologic", "CESC"), ("gynecologic", "OV"), ("gynecologic", "UCEC"), ("gynecologic", "UCS"),
    ("hematopoietic", "DLBC"), ("hematopoietic", "THYM"),
    ("melanocytic", "SKCM"), ("melanocytic", "UVM"), 
    ("pulmonary", "LUAD"), ("pulmonary", "LUSC"), ("pulmonary", "MESO"), 
    ("urinary", "BLCA"), ("urinary", "KICH"), ("urinary", "KIRC"), ("urinary", "KIRP"), 
    ("prostate", "PRAD"), ("prostate", "TGCT"), 
    ("endocrine", "ACC"), ("endocrine", "PCPG"), ("endocrine", "THCA"), 
    ("liver", "CHOL"), ("liver", "LIHC"), ("liver", "PAAD"), 
]


SITE2LABEL = {site: idx for idx, site in enumerate(sorted(set(site for site, _ in site_subtype)))}

SUBTYPE2LABEL = {subtype: idx for idx, subtype in enumerate(sorted(set(subtype for _, subtype in site_subtype)))}



def train_and_evaluate_kfold(data, pooling, task, encode_name, embedding_dim=2048,
                              batch_size=4, threshold=0.1, epochs=10, lr=1e-3,
                              log_path="experiment_log.txt", k_folds=5):
    assert task in ['site', 'subtype'], "Task must be either 'site' or 'subtype'."
    num_classes = 10 if task == 'site' else 29
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset and labels
    inputs = []
    targets = []
    for sample in data:
        feature = [{
                    "location":d['location'],
                    "embedding":d['embedding']
                } for d in sample]

        if task == "site":
            label = SITE2LABEL[sample[0][task]]
        elif task == "subtype":
            label = SUBTYPE2LABEL[sample[0][task]]

        inputs.append(feature)
        targets.append(int(label))
    
    label_count = Counter(targets)
    print(label_count)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_train_accs = []
    all_val_accs = []
    all_fold_losses = []
    task_val_accs_dict = {}

    # 每类验证统计初始化
    all_val_class_correct = [0 for _ in range(num_classes)]
    all_val_class_total = [0 for _ in range(num_classes)]

    for fold, (train_idx, val_idx) in enumerate(skf.split(inputs, targets)):

        classifier = WSIGraphClassifier(embedding_dim=embedding_dim, num_classes=num_classes).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_inputs = [inputs[i] for i in train_idx]
        train_labels = [targets[i] for i in train_idx]
        val_inputs = [inputs[i] for i in val_idx]
        val_labels = [targets[i] for i in val_idx]

        print(f"\n🚀 Starting Fold {fold+1}/{k_folds}...")

        fold_train_accs = []
        fold_val_accs = []
        fold_losses = []

        with tqdm(range(1, epochs + 1), desc=f"Fold {fold+1}/{k_folds}", ascii=True) as pbar:
            for epoch in pbar:

                classifier.train()
                total_loss = 0
                correct = 0
                total = 0

                for i in range(0, len(train_inputs), batch_size):
                    batch_data = train_inputs[i:i + batch_size]
                    batch_labels = train_labels[i:i + batch_size]

                    pools = []
                    for j, node_list in enumerate(batch_data):
                        embeddings = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list])

                        if pooling == "max":
                            pooled, _ = embeddings.max(dim=0)
                        elif pooling == "mean":
                            pooled = embeddings.mean(dim=0)

                        pools.append(pooled)

                    batch_pool = torch.stack(pools, dim=0).to(device)
                    logits_tensor = classifier.forward(batch_pool)
                    labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

                    loss = criterion(logits_tensor, labels_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    preds = logits_tensor.argmax(dim=1)
                    correct += (preds == labels_tensor).sum().item()
                    total += len(batch_labels)

                train_acc = correct / total
                avg_loss = total_loss / (len(train_inputs) // batch_size + 1)

                # Validation
                classifier.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(0, len(val_inputs), batch_size):
                        batch_data = val_inputs[i:i + batch_size]
                        batch_labels = val_labels[i:i + batch_size]

                        pools = []
                        for j, node_list in enumerate(batch_data):
                            embeddings = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list])

                            if pooling == "max":
                                pooled, _ = embeddings.max(dim=0)
                            elif pooling == "mean":
                                pooled = embeddings.mean(dim=0)

                            pools.append(pooled)

                        batch_pool = torch.stack(pools, dim=0).to(device)
                        logits_tensor = classifier.forward(batch_pool)
                        labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

                        preds = logits_tensor.argmax(dim=1)
                        correct += (preds == labels_tensor).sum().item()
                        total += len(batch_labels)

                val_acc = correct / total

                fold_train_accs.append(train_acc)
                fold_val_accs.append(val_acc)
                fold_losses.append(avg_loss)

                with open(log_path, "a") as f:
                    f.write(f"[{datetime.now()}] Fold {fold+1}/{k_folds} - Epoch {epoch}/{epochs} - "
                            f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                            f"Task: {task}, Batch Size: {batch_size}, LR: {lr}\n")

                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": f"{avg_loss:.4f}",
                    "Train Acc": f"{train_acc:.4f}",
                    "Val Acc": f"{val_acc:.4f}"
                })

        # 存储 fold 的信息
        all_train_accs.append(fold_train_accs)
        all_val_accs.append(fold_val_accs)
        all_fold_losses.append(fold_losses)

        # 更新每个任务的 val acc 记录
        if task not in task_val_accs_dict:
            task_val_accs_dict[task] = []
        task_val_accs_dict[task].append(fold_val_accs)

        # === 每 fold 结束后再统计每类 val acc ===
        classifier.eval()
        with torch.no_grad():
            for i in range(0, len(val_inputs), batch_size):
                batch_data = val_inputs[i:i + batch_size]
                batch_labels = val_labels[i:i + batch_size]

                pools = []
                for j, node_list in enumerate(batch_data):
                    embeddings = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in node_list])

                    if pooling == "max":
                        pooled, _ = embeddings.max(dim=0)
                    elif pooling == "mean":
                        pooled = embeddings.mean(dim=0)

                    pools.append(pooled)

                batch_pool = torch.stack(pools, dim=0).to(device)
                logits_tensor = classifier.forward(batch_pool)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

                preds = logits_tensor.argmax(dim=1)
                for label, pred in zip(labels_tensor, preds):
                    label = label.item()
                    if 0 <= label < num_classes:
                        all_val_class_total[label] += 1
                        if pred.item() == label:
                            all_val_class_correct[label] += 1

    # === 所有 fold 结束后 ===

    avg_train_acc = sum([sum(acc) for acc in all_train_accs]) / (k_folds * epochs)
    avg_val_acc = sum([sum(acc) for acc in all_val_accs]) / (k_folds * epochs)
    avg_loss = sum([sum(loss) for loss in all_fold_losses]) / (k_folds * epochs)

    with open(log_path, "a") as f:
        f.write(f"\n[{datetime.now()}] === Final Results ===\n")
        f.write(f"Encode Backbone: {encode_name}\n")
        f.write(f"WSI model: ABMIL\n")
        f.write(f"Task: {task}\n")
        f.write(f"Batch Size: {batch_size}, LR: {lr}, Epochs: {epochs}, K-Folds: {k_folds}\n")
        f.write(f"Average Train Accuracy: {avg_train_acc:.4f}\n")
        f.write(f"Using Encode: {encode_name}\n")
        f.write(f"Average Validation Accuracy (all tasks): {avg_val_acc:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")

        # 写入任务级平均 val acc
        for task_name, val_accs_list in task_val_accs_dict.items():
            total = sum([sum(fold_accs) for fold_accs in val_accs_list])
            count = sum([len(fold_accs) for fold_accs in val_accs_list])
            avg_task_val_acc = total / count
            f.write(f"Task: {task_name} - Avg Val Acc: {avg_task_val_acc:.4f}\n")

        # 写入每类 val acc
        f.write("\n=== Per-Class Validation Accuracy ===\n")
        for i in range(num_classes):
            correct = all_val_class_correct[i]
            total = all_val_class_total[i]
            if total > 0:
                acc = correct / total
                f.write(f"Class {i} - Val Acc: {acc:.4f} ({correct}/{total})\n")
            else:
                f.write(f"Class {i} - Val Acc: N/A (no samples)\n")









def load_json_data(folder):
    data = []
    for fname in tqdm(os.listdir(folder), desc=f"Loading data from {folder}."):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname), 'r') as f:
                sample = json.load(f)
                data.append(sample)
    return data

def main(args):
    data_folder = f"a5037_course_work/data/embeddings/{args.encode_name}"
    data = load_json_data(data_folder)

    log_path = args.log_path or f"a5037_course_work/experiments/results/{args.encode_name}_{args.pooling}_{args.task}_results.txt"

    if args.encode_name == "resnet":
        embedding_dim = 2048
    elif args.encode_name == "vit":
        embedding_dim = 768
    elif args.encode_name == "dino":
        embedding_dim = 768
    elif args.encode_name == "uni":
        embedding_dim = 1024

    train_and_evaluate_kfold(
        data=data,
        pooling=args.pooling,
        task=args.task,
        encode_name=args.encode_name,
        embedding_dim=embedding_dim,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_path=log_path,
        k_folds=args.k_folds
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-fold classification on WSI embeddings.")
    parser.add_argument("--encode_name", type=str, choices=["vit", "dino", "resnet", "uni"], required=True)
    parser.add_argument("--pooling", type=str, choices=["mean", "max"], required=True)
    parser.add_argument("--task", type=str, choices=["site", "subtype"], required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--log_path", type=str, default=None)

    args = parser.parse_args()
    main(args)

    # python3 a5037_course_work/experiments/classification_basic.py --encode_name resnet --pooling max --task site 