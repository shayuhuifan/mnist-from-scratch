import mynn as nn
import numpy as np
from struct import unpack
import gzip

test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs     = test_imgs / test_imgs.max()
test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)

configs = ['SGD', 'MomentGD', 'SGD_WD', 'MomentGD_WD']

print(f"{'Model':<20} {'Config':<15} {'Accuracy':>8}")
print('-' * 45)

for cfg in configs:
    mlp_model = nn.models.Model_MLP()
    mlp_model.load_model(f'./best_models/mlp_{cfg}.pickle')
    mlp_acc = nn.metric.accuracy(mlp_model(test_imgs), test_labs)
    print(f"{'MLP':<20} {cfg:<15} {mlp_acc:>8.5f}")

for cfg in configs:
    cnn_model = nn.models.Model_CNN()
    cnn_model.load_model(f'./best_models/cnn_{cfg}.pickle')
    cnn_acc = nn.metric.accuracy(cnn_model(test_imgs_cnn), test_labs)
    print(f"{'CNN':<20} {cfg:<15} {cnn_acc:>8.5f}")
