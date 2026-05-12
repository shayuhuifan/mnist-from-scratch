import mynn as nn
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import gzip


configs = ['SGD', 'MomentGD', 'SGD_WD', 'MomentGD_WD']


def confusion_matrix(preds, labels, num_classes=10):
    pred_labels = np.argmax(preds, axis=-1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, pred_labels):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    return cm.diagonal() / cm.sum(axis=1)


def plot_confusion_matrix(ax, cm, title):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_ylabel('True', fontsize=7)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')
    return im


# ============================================================
# 加载测试集
# ============================================================
test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs     = test_imgs / test_imgs.max()
test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)

# ============================================================
# 加载全部模型
# ============================================================
mlp_models = {}
cnn_models = {}
for cfg in configs:
    m = nn.models.Model_MLP()
    m.load_model(f'./best_models/mlp_{cfg}.pickle')
    mlp_models[cfg] = m

    c = nn.models.Model_CNN()
    c.load_model(f'./best_models/cnn_{cfg}.pickle')
    cnn_models[cfg] = c

# ============================================================
# Figure 1a: MLP Confusion Matrix  —  1 行 × 4 列
# ============================================================
fig1a, axes = plt.subplots(1, 4, figsize=(20, 6))
fig1a.suptitle('MLP  Confusion Matrices', fontsize=13)

for j, cfg in enumerate(configs):
    im = plot_confusion_matrix(axes[j],
                               confusion_matrix(mlp_models[cfg](test_imgs), test_labs),
                               f'[{cfg}]')
    plt.colorbar(im, ax=axes[j])

fig1a.tight_layout()

# ============================================================
# Figure 1b: CNN Confusion Matrix  —  1 行 × 4 列
# ============================================================
fig1b, axes = plt.subplots(1, 4, figsize=(20, 6))
fig1b.suptitle('CNN  Confusion Matrices', fontsize=13)

for j, cfg in enumerate(configs):
    im = plot_confusion_matrix(axes[j],
                               confusion_matrix(cnn_models[cfg](test_imgs_cnn), test_labs),
                               f'[{cfg}]')
    plt.colorbar(im, ax=axes[j])

fig1b.tight_layout()

# ============================================================
# Figure 1c / 1d: Per-class Accuracy  —  MLP 和 CNN 各一张
# ============================================================
digits    = np.arange(10)
bar_width = 0.18
offsets   = (np.arange(len(configs)) - (len(configs) - 1) / 2) * bar_width
colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for model_dict, imgs_input, title_prefix in [
        (mlp_models, test_imgs,     'MLP'),
        (cnn_models, test_imgs_cnn, 'CNN'),
]:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'{title_prefix}  Per-class Accuracy', fontsize=13)

    for k, (cfg, color, offset) in enumerate(zip(configs, colors, offsets)):
        cm  = confusion_matrix(model_dict[cfg](imgs_input), test_labs)
        acc = per_class_accuracy(cm)
        bars = ax.bar(digits + offset, acc, width=bar_width,
                      label=cfg, color=color)
        for bar, v in zip(bars, acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(digits)
    ax.set_xticklabels([f'Digit {d}' for d in digits])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.08)
    ax.legend()
    fig.tight_layout()

# ============================================================
# Figure 2: MLP 第一层权重  —  4 行(configs) × 10 列(neurons)
# ============================================================
fig2, axes = plt.subplots(4, 10, figsize=(14, 6))
fig2.suptitle('MLP  Layer-1 Weights (first 10 / 600 neurons)')

for i, cfg in enumerate(configs):
    W1 = mlp_models[cfg].layers[0].params['W']   # (784, 600)
    for j in range(10):
        ax = axes[i, j]
        ax.matshow(W1.T[j].reshape(28, 28), cmap='bwr')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[i, 0].set_ylabel(cfg, fontsize=7)

fig2.tight_layout()

# ============================================================
# Figure 3: MLP 第二层权重矩阵  —  1 行 × 4 列
# ============================================================
fig3, axes = plt.subplots(1, 4, figsize=(16, 5))
fig3.suptitle('MLP  Layer-2 Weights (600 × 10)')

for j, cfg in enumerate(configs):
    W2 = mlp_models[cfg].layers[2].params['W']   # (600, 10)
    im = axes[j].matshow(W2, cmap='bwr', aspect='auto')
    axes[j].set_title(cfg, fontsize=8)
    axes[j].set_xlabel('Class', fontsize=7)
    axes[j].set_ylabel('Hidden unit', fontsize=7)
    plt.colorbar(im, ax=axes[j])

fig3.tight_layout()

# ============================================================
# Figure 4: CNN Conv1 卷积核  —  4 行(configs) × 8 列(filters)
# ============================================================
fig4, axes = plt.subplots(4, 8, figsize=(12, 6))
fig4.suptitle('CNN  Conv1 Filters (8 × 3×3 per config)')

for i, cfg in enumerate(configs):
    W = cnn_models[cfg].layers[0].params['W']    # (8, 1, 3, 3)
    for j in range(8):
        ax = axes[i, j]
        ax.matshow(W[j, 0], cmap='bwr')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[i, 0].set_ylabel(cfg, fontsize=7)

fig4.tight_layout()

# ============================================================
# Figure 5: CNN 最终 FC 层权重矩阵  —  1 行 × 4 列
# ============================================================
fig5, axes = plt.subplots(1, 4, figsize=(16, 5))
fig5.suptitle('CNN  FC Layer-2 Weights (128 × 10)')

for j, cfg in enumerate(configs):
    W = cnn_models[cfg].layers[6].params['W']    # (128, 10)
    im = axes[j].matshow(W, cmap='bwr', aspect='auto')
    axes[j].set_title(cfg, fontsize=8)
    axes[j].set_xlabel('Class', fontsize=7)
    axes[j].set_ylabel('Hidden unit', fontsize=7)
    plt.colorbar(im, ax=axes[j])

fig5.tight_layout()

plt.show()
