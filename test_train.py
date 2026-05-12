import mynn as nn

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

np.random.seed(309)

train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs_cnn = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs_cnn = valid_imgs.reshape(-1, 1, 28, 28)

# ============================================================
# 消融实验配置
# ============================================================
configs = [
    {'name': 'SGD',           'use_moment': False, 'use_wd': False},
    {'name': 'MomentGD',      'use_moment': True,  'use_wd': False},
    {'name': 'SGD_WD',        'use_moment': False, 'use_wd': True},
    {'name': 'MomentGD_WD',   'use_moment': True,  'use_wd': True},
]

def make_optimizer(use_moment, model):
    if use_moment:
        return nn.optimizer.MomentGD(init_lr=0.01, model=model, mu=0.9)
    else:
        return nn.optimizer.SGD(init_lr=0.06, model=model)

def configure_weight_decay(model, use_wd, lambda_val=1e-4):
    for layer in model.layers:
        if layer.optimizable:
            layer.weight_decay = use_wd
            layer.weight_decay_lambda = lambda_val if use_wd else 0.0

mlp_runners = []
cnn_runners = []

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Training MLP  [{cfg['name']}]")
    print('='*50)

    # ---- MLP ----
    mlp_model   = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU')
    configure_weight_decay(mlp_model, cfg['use_wd'])
    opt_mlp     = make_optimizer(cfg['use_moment'], mlp_model)
    sched_mlp   = nn.lr_scheduler.MultiStepLR(optimizer=opt_mlp, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn_mlp = nn.op.MultiCrossEntropyLoss(model=mlp_model, max_classes=train_labs.max()+1)
    runner_mlp  = nn.runner.RunnerM(mlp_model, opt_mlp, nn.metric.accuracy, loss_fn_mlp, scheduler=sched_mlp)
    runner_mlp.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                     num_epochs=2, log_iters=100, save_dir='./best_models',
                     save_name=f'mlp_{cfg["name"]}.pickle')
    mlp_runners.append((cfg['name'], runner_mlp))

    print(f"\n{'='*50}")
    print(f"Training CNN  [{cfg['name']}]")
    print('='*50)

    # ---- CNN ----
    cnn_model   = nn.models.Model_CNN(
        conv_configs=[
            {'in_channels': 1,  'out_channels': 8,  'kernel_size': 3},
            {'in_channels': 8,  'out_channels': 16, 'kernel_size': 3},
        ],
        fc_configs=[9216, 128, 10],
        act_func='ReLU',
    )
    configure_weight_decay(cnn_model, cfg['use_wd'])
    opt_cnn     = make_optimizer(cfg['use_moment'], cnn_model)
    sched_cnn   = nn.lr_scheduler.MultiStepLR(optimizer=opt_cnn, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn_cnn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)
    runner_cnn  = nn.runner.RunnerM(cnn_model, opt_cnn, nn.metric.accuracy, loss_fn_cnn, scheduler=sched_cnn)
    runner_cnn.train([train_imgs_cnn, train_labs], [valid_imgs_cnn, valid_labs],
                     num_epochs=2, log_iters=100, save_dir='./best_models',
                     save_name=f'cnn_{cfg["name"]}.pickle')
    cnn_runners.append((cfg['name'], runner_cnn))

# ============================================================
# 消融对比图：每个模型一张图，dev_loss 和 dev_score 各一列
# ============================================================
colors     = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.', ':']

def plot_ablation(runners, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} Ablation Study')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('dev loss')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('dev score')

    for (name, runner), color, ls in zip(runners, colors, linestyles):
        dev_x = [i * runner.log_iters for i in range(len(runner.dev_scores))]
        axes[0].plot(dev_x, runner.dev_loss,   color=color, linestyle=ls, label=name)
        axes[1].plot(dev_x, runner.dev_scores, color=color, linestyle=ls, label=name)

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

plot_ablation(mlp_runners, 'MLP')
plot_ablation(cnn_runners, 'CNN')

plt.show()
