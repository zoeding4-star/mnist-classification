"""
main.py — 主程序入口
========================
一键运行所有对照实验，自动保存图表和结果

实验设计:
  1. 网络结构对比: MLP_Shallow vs MLP_Deep vs CNN_Simple vs CNN_Better
  2. 损失函数对比: CrossEntropy vs CrossEntropy+LabelSmoothing (CNN_Better)
  3. 超参数对比 (learning rate, batch size)
  4. 正则化对比: Dropout / L2 / 无正则
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import set_seed, load_mnist_local, get_dataloaders, visualize_samples, visualize_class_distribution
from models import create_model, count_params
from train import train_model, plot_training_history
from eval import evaluate, print_classification_report, plot_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim


RESULTS = []  # 全局结果列表


def run_exp(config, train_loader, val_loader, test_loader, device):
    """
    运行单个实验配置
    config dict:
        name, model_type, loss_fn, lr, batch_size, dropout, weight_decay, epochs, patience
    """
    name = config["name"]
    print(f"\n{"#"*70}")
    print(f"# 实验: {name}")
    print(f"# 模型={config['model_type']} 损失={config['loss_fn']} "
          f"lr={config['lr']} bs={config['batch_size']} "
          f"drop={config['dropout']} wd={config['weight_decay']}")
    print(f"{"#"*70}")

    model = create_model(config["model_type"], dropout_rate=config["dropout"]).to(device)
    params = count_params(model)
    print(f"  参数量: {params:,}")

    if config["loss_fn"] == "ce":
        criterion = nn.CrossEntropyLoss()
    elif config["loss_fn"] == "label_smooth":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"未知损失函数: {config['loss_fn']}")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    history, best_val = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=config["epochs"], device=device,
        patience=config["patience"],
        save_path=f"weights/{name}_best.pth",
        model_name=name,
    )

    # 绘制训练曲线
    curve_path = f"../report/figures/curve_{name}.pdf"
    plot_training_history(history, save_path=curve_path, title=name)

    # 测试
    model.load_state_dict(torch.load(f"weights/{name}_best.pth", map_location=device, weights_only=True))
    test_acc, test_loss, all_preds, all_labels = evaluate(model, test_loader, criterion, device, name)

    # 分类报告 & 混淆矩阵
    print_classification_report(all_labels, all_preds, save_path=f"../report/{name}_report.txt")
    cm_path = f"../report/figures/cm_{name}.pdf"
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path, title=name)

    result = {
        "name": name, "model": config["model_type"], "params": params,
        "loss_fn": config["loss_fn"], "lr": config["lr"], "batch_size": config["batch_size"],
        "dropout": config["dropout"], "weight_decay": config["weight_decay"],
        "val_acc": best_val, "test_acc": test_acc, "test_loss": test_loss,
    }
    RESULTS.append(result)
    return result


def print_summary():
    print("\n" + "=" * 100)
    print("  对照实验汇总结果")
    print("=" * 100)
    header = f"{'实验名称':<28} {'模型':<14} {'损失':<14} {'LR':<10} {'BS':<6} {'Drop':<6} {'L2':<10} {'Params':<8} {'Val Acc':<10} {'Test Acc':<10}"
    print(header)
    print("-" * 116)
    for r in RESULTS:
        print(f"{r['name']:<28} {r['model']:<14} {r['loss_fn']:<14} "
              f"{r['lr']:<10} {r['batch_size']:<6} {r['dropout']:<6} {r['weight_decay']:<10} "
              f"{r['params']:<8,} {r['val_acc']:.4f}     {r['test_acc']:.4f}")

    # 保存汇总
    summary_path = "../report/experiment_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("对照实验汇总结果\n" + "="*70 + "\n")
        f.write(header + "\n" + "-"*116 + "\n")
        for r in RESULTS:
            f.write(f"{r['name']:<28} {r['model']:<14} {r['loss_fn']:<14} "
                    f"{r['lr']:<10} {r['batch_size']:<6} {r['dropout']:<6} {r['weight_decay']:<10} "
                    f"{r['params']:<8,} {r['val_acc']:.4f}     {r['test_acc']:.4f}\n")
    print(f"\n汇总表 -> {summary_path}")


def main():
    t_start = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    train_ds, val_ds, test_ds = load_mnist_local()
    visualize_samples(train_ds)
    visualize_class_distribution(train_ds, val_ds, test_ds)

    # 根据设备选择epoch数 (CPU慢, GPU快)
    epochs_cnn = 8 if device.type == "cpu" else 30
    epochs_mlp = 15 if device.type == "cpu" else 30
    patience = 4 if device.type == "cpu" else 7

    # ====== 实验组1: 网络结构对比 (固定: CE, lr=1e-3, bs=64, no reg) ======
    print("\n" + "="*60)
    print("实验组1: 网络结构对比")
    print("="*60)
    train_l, val_l, test_l = get_dataloaders(train_ds, val_ds, test_ds, batch_size=64)

    struct_cfg = {"loss_fn": "ce", "lr": 1e-3, "batch_size": 64,
                  "dropout": 0.0, "weight_decay": 0.0}

    for model_type, ep in [("MLP_Shallow", epochs_mlp),
                            ("MLP_Deep", epochs_mlp),
                            ("CNN_Simple", epochs_cnn),
                            ("CNN_Better", epochs_cnn)]:
        run_exp({**struct_cfg, "name": f"exp1_{model_type}",
                 "model_type": model_type, "epochs": ep, "patience": patience},
                train_l, val_l, test_l, device)

    # ====== 实验组2: 损失函数对比 (CNN_Better固定, lr=1e-3, bs=64) ======
    print("\n" + "="*60)
    print("实验组2: 损失函数对比")
    print("="*60)
    for loss_fn in ["ce", "label_smooth"]:
        run_exp({"name": f"exp2_{loss_fn}", "model_type": "CNN_Better",
                 "loss_fn": loss_fn, "lr": 1e-3, "batch_size": 64,
                 "dropout": 0.5, "weight_decay": 0.0,
                 "epochs": epochs_cnn, "patience": patience},
                train_l, val_l, test_l, device)

    # ====== 实验组3a: 学习率对比 (CNN_Better, CE, bs=64) ======
    print("\n" + "="*60)
    print("实验组3a: 学习率对比")
    print("="*60)
    for lr in [1e-2, 1e-3, 1e-4]:
        run_exp({"name": f"exp3a_lr{lr}", "model_type": "CNN_Simple",
                 "loss_fn": "ce", "lr": lr, "batch_size": 64,
                 "dropout": 0.0, "weight_decay": 0.0,
                 "epochs": epochs_cnn, "patience": patience},
                train_l, val_l, test_l, device)

    # ====== 实验组3b: batch_size对比 (CNN_Simple, CE, lr=1e-3) ======
    print("\n" + "="*60)
    print("实验组3b: Batch Size对比")
    print("="*60)
    for bs in [16, 64, 256]:
        tr_l, va_l, te_l = get_dataloaders(train_ds, val_ds, test_ds, batch_size=bs)
        run_exp({"name": f"exp3b_bs{bs}", "model_type": "CNN_Simple",
                 "loss_fn": "ce", "lr": 1e-3, "batch_size": bs,
                 "dropout": 0.0, "weight_decay": 0.0,
                 "epochs": epochs_cnn, "patience": patience},
                tr_l, va_l, te_l, device)

    # ====== 实验组4: 正则化对比 (CNN_Better, CE, lr=1e-3, bs=64) ======
    print("\n" + "="*60)
    print("实验组4: 正则化对比")
    print("="*60)
    reg_cfg = {"loss_fn": "ce", "lr": 1e-3, "batch_size": 64,
               "epochs": epochs_cnn, "patience": patience}
    for dropout, wd, suffix in [(0.0, 0.0, "none"),
                                 (0.5, 0.0, "drop05"),
                                 (0.0, 1e-4, "l2_1e4"),
                                 (0.5, 1e-4, "both")]:
        run_exp({**reg_cfg, "name": f"exp4_{suffix}",
                 "model_type": "CNN_Better",
                 "dropout": dropout, "weight_decay": wd},
                train_l, val_l, test_l, device)

    # ====== 汇总 ======
    print_summary()
    total_time = time.time() - t_start
    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
