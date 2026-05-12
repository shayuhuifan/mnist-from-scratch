from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        # 为每个可优化层的每个参数初始化速度缓冲区
        self.velocity = {}
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[id(layer)] = {
                    key: np.zeros_like(param)
                    for key, param in layer.params.items()
                }

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                v = self.velocity[id(layer)]
                for key in layer.params.keys():
                    # v = mu * v + grad
                    v[key] = self.mu * v[key] + layer.grads[key]
                    layer.params[key] -= self.init_lr * v[key]