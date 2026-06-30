"""
models.py — 3种以上网络结构定义
==================================
1. MLP_Shallow:  1层隐藏层 (784->256->10)          浅层MLP
2. MLP_Deep:     3层隐藏层 (784->512->256->128->10) 深层MLP+BN
3. CNN_Simple:   2层卷积 + 2层FC                    简单CNN
4. CNN_Better:   3层卷积+BN+Dropout+GlobalAvg       改进CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 浅层MLP ----------
class MLP_Shallow(nn.Module):
    """1层隐藏层: 784 -> 256 -> 10"""
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ---------- 2. 深层MLP ----------
class MLP_Deep(nn.Module):
    """3层隐藏层: 784 -> 512 -> 256 -> 128 -> 10 + BN"""
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        layers = []
        dims = [784, 512, 256, 128]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ---------- 3. 简单CNN (2层卷积) ----------
class CNN_Simple(nn.Module):
    """
    结构: Conv(1->16,3) -> Pool -> Conv(16->32,3) -> Pool -> FC(32*7*7->128) -> FC(10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------- 4. 改进CNN (3层卷积+BN+Dropout) ----------

class CNN_Better(nn.Module):
    """
    改进CNN (轻量版): Conv2个 + BN + Dropout + 更少通道
    Conv(1->16,3)+BN+ReLU -> Conv(16->32,3)+BN+ReLU -> Pool(2)
    Dropout -> FC(32*14*14->128) -> Dropout -> FC(10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ====== 模型工厂 ======
MODEL_REGISTRY = {
    "MLP_Shallow": MLP_Shallow,
    "MLP_Deep":    MLP_Deep,
    "CNN_Simple":  CNN_Simple,
    "CNN_Better":  CNN_Better,
}

def create_model(model_type, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}, options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        model = create_model(name)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        print(f"{name:20s} | params={count_params(model):>8,d} | out={list(out.shape)}")
