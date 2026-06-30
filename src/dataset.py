"""
dataset.py — MNIST数据集加载与预处理
功能: 解析本地gz文件, 归一化, 划分(55000/5000/10000), 可视化, 固定种子
"""
import os, gzip, struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_idx_images(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"图像魔数错误: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data

def parse_idx_labels(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"标签魔数错误: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

class MNISTLocal(Dataset):
    def __init__(self, images, labels, transform=True):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), torch.tensor(label)

def load_mnist_local(data_dir=r"D:\mnist", val_size=5000):
    paths = {
        "train_img": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_lbl": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_img":  os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_lbl":  os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    }
    for p in paths.values():
        assert os.path.exists(p), f"文件不存在: {p}"
    print("[dataset] 解析训练集...")
    train_images = parse_idx_images(paths["train_img"])
    train_labels = parse_idx_labels(paths["train_lbl"])
    print("[dataset] 解析测试集...")
    test_images = parse_idx_images(paths["test_img"])
    test_labels = parse_idx_labels(paths["test_lbl"])
    indices = np.random.RandomState(SEED).permutation(len(train_labels))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    print(f"[dataset] 划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_labels)}")
    return (
        MNISTLocal(train_images[train_idx], train_labels[train_idx]),
        MNISTLocal(train_images[val_idx],   train_labels[val_idx]),
        MNISTLocal(test_images, test_labels),
    )

def get_dataloaders(train_ds, val_ds, test_ds, batch_size=64, num_workers=0):
    return (
        DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers),
    )

def visualize_samples(dataset, num_samples=10, save_path="../report/figures/samples.pdf"):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"Label: {label.item()}")
        axes[i].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[dataset] 样本可视化 -> {save_path}")

def visualize_class_distribution(train_ds, val_ds, test_ds,
                                  save_path="../report/figures/class_dist.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, ds) in zip(axes, [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]):
        labels = [ds[i][1].item() for i in range(len(ds))]
        ax.hist(labels, bins=range(11), align='left', rwidth=0.8, alpha=0.7, color='steelblue')
        ax.set_title(f"{name} (n={len(ds)})")
        ax.set_xlabel("Digit"); ax.set_ylabel("Count"); ax.set_xticks(range(10))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[dataset] 类别分布 -> {save_path}")

if __name__ == "__main__":
    set_seed()
    train_ds, val_ds, test_ds = load_mnist_local()
    img, label = train_ds[0]
    print(f"样本 shape={img.shape}, label={label}")
    visualize_samples(train_ds)
    visualize_class_distribution(train_ds, val_ds, test_ds)
