"""
eval.py — 测试集评估模块
功能: 测试集评估, 分类报告, 混淆矩阵热力图
"""
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, test_loader, criterion, device, model_name="model"):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    avg_loss = test_loss / total
    print(f"\n[{model_name}] 测试集: Loss={avg_loss:.4f} Acc={accuracy:.4f} ({correct}/{total})")
    return accuracy, avg_loss, np.array(all_preds), np.array(all_labels)


def print_classification_report(all_labels, all_preds, save_path=None):
    target_names = [str(i) for i in range(10)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\n分类报告:\n" + report)
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report\n" + "="*60 + "\n" + report)
        print(f"[eval] 分类报告 -> {save_path}")


def plot_confusion_matrix(all_labels, all_preds, save_path="../report/figures/confusion_matrix.pdf",
                           title="Confusion Matrix"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[eval] 混淆矩阵 -> {save_path}")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"  数字 {i}: {acc:.4f}")
    return class_acc
