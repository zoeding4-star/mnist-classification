"""
train.py — 训练与验证逻辑
功能: 单epoch训练/验证, 早停机制, 最优权重保存, 损失/准确率曲线绘制
"""
import os, time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=30, device="cpu", patience=7,
                save_path="weights/best_model.pth", model_name="model"):
    """
    完整训练流程: 含早停、权重保存、指标记录
    返回: (history, best_val_acc)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  [{model_name}] 开始训练 | device={device} | epochs={num_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [早停] {patience}个epoch未提升, 停止于Epoch {epoch}")
                break

    print(f"{'='*60}")
    print(f"  [{model_name}] 完成 | 最优Val Acc={best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*60}")
    return history, best_val_acc


def plot_training_history(history, save_path="../report/figures/training_curve.pdf",
                          title="Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], 'b-', label="Train")
    axes[0].plot(epochs, history["val_loss"], 'r-', label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], 'b-', label="Train")
    axes[1].plot(epochs, history["val_acc"], 'r-', label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Acc"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[train] 训练曲线 -> {save_path}")
