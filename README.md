# MNIST 手写数字识别

## 项目简介
基于PyTorch框架完成MNIST 0-9多分类任务。对比多种网络结构、损失函数、超参数的影响。

## 网络结构 (4种)
| 模型 | 结构 | 参数量 |
|------|------|--------|
| MLP_Shallow | 784→256→10 | ~100K |
| MLP_Deep | 784→512→256→128→10 + BN | ~577K |
| CNN_Simple | Conv*2 + FC*2 | ~422K |
| CNN_Better | Conv*3+BN + FC + Dropout | ~391K |

## 项目结构
```
mnist-classification/
├── src/
│   ├── dataset.py    # 数据加载与预处理
│   ├── models.py     # 4种网络结构
│   ├── train.py      # 训练与早停
│   ├── eval.py       # 评估与可视化
│   └── main.py       # 主程序入口
├── report/
│   ├── main.tex      # LaTeX实验报告
│   └── figures/      # 实验图表
├── weights/          # 模型权重（已gitignore）
├── .gitignore
└── README.md
```

## 运行
```bash
cd src
python main.py
```
