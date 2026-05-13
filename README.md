# mnist-from-scratch

从零实现 MLP 与 CNN（仅用 NumPy），在 MNIST 手写数字数据集上完成训练、消融实验与可视化分析。

## 文件结构

```
codes/
├── mynn/                    # 自实现神经网络库
│   ├── __init__.py
│   ├── op.py                # 基础算子：Linear、conv2D、ReLU、MultiCrossEntropyLoss
│   ├── models.py            # 模型定义：Model_MLP、Model_CNN
│   ├── optimizer.py         # 优化器：SGD、MomentGD
│   ├── lr_scheduler.py      # 学习率调度：MultiStepLR
│   ├── runner.py            # 训练流程封装：RunnerM
│   └── metric.py            # 评估指标：accuracy
├── test_train.py            # 训练脚本（消融实验）
├── test_model.py            # 测试脚本（加载模型评估准确率）
└── weight_visualization.py  # 可视化脚本（混淆矩阵、权重图等）
```

## 环境依赖

```
Python >= 3.8
numpy
matplotlib
```

无需 PyTorch / TensorFlow，所有前向传播与反向传播均基于 NumPy 手动实现。

## 模型结构

**MLP**
```
Linear(784, 600) → ReLU → Linear(600, 10)
```

**CNN**
```
Conv(1, 8, 3) → ReLU → Conv(8, 16, 3) → ReLU → Linear(9216, 128) → ReLU → Linear(128, 10)
```

两种模型均使用 He 初始化，损失函数为带 Softmax 的交叉熵。

## 使用方法

### 1. 训练（消融实验）

```bash
python test_train.py
```

对 MLP 和 CNN 各运行四种配置的消融实验：

| 配置 | 优化器 | Weight Decay |
|---|---|---|
| SGD | SGD (lr=0.06) | ✗ |
| MomentGD | Momentum SGD (lr=0.01, μ=0.9) | ✗ |
| SGD_WD | SGD | ✓ (λ=1e-4) |
| MomentGD_WD | Momentum SGD | ✓ (λ=1e-4) |

训练完成后模型自动保存至 `best_models/`（不含于版本控制）。

### 2. 测试

```bash
python test_model.py
```

加载 `best_models/` 中所有模型，在测试集上输出各配置的准确率。

### 3. 可视化

```bash
python weight_visualization.py
```

加载 `best_models/` 中所有模型，生成混淆矩阵、逐类别准确率、权重热图等图表。

## 实验结果

| 配置 | MLP Accuracy | CNN Accuracy |
|---|---|---|
| SGD | 0.9462 | 0.9840 |
| MomentGD | 0.9585 | 0.9855 |
| SGD_WD | 0.9445 | 0.9834 |
| MomentGD_WD | 0.9591 | 0.9845 |
