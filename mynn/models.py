from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_configs=None, fc_configs=None, act_func=None, lambda_list=None):
        self.conv_configs = conv_configs
        self.fc_configs = fc_configs
        self.act_func = act_func
        self.layers = []

        if conv_configs is not None and act_func is not None:
            
            for i, config in enumerate(conv_configs):
                layer = conv2D(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', 1),
                    padding=config.get('padding', 0)
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                
                self.layers.append(layer)
                
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError

            if fc_configs is not None:
                for i in range(len(fc_configs) - 1):
                    fc_layer = Linear(
                        in_dim=fc_configs[i],
                        out_dim=fc_configs[i + 1]
                    )
                    if lambda_list is not None:
                        fc_layer.weight_decay = True
                        fc_layer.weight_decay_lambda = lambda_list[len(conv_configs) + i]
                    
                    self.layers.append(fc_layer)
                    
                    if i < len(fc_configs) - 2:
                        if act_func == 'ReLU':
                            self.layers.append(ReLU())
                        elif act_func == 'Logistic':
                            raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.conv_configs is not None and self.act_func is not None, \
            'Model has not initialized yet.'
        
        outputs = X  # shape: (batch, channels, H, W)
        self.flatten_input_shape = None
        
        for layer in self.layers:
            if isinstance(layer, Linear) and len(outputs.shape) == 4:
                self.flatten_input_shape = outputs.shape
                outputs = outputs.reshape(outputs.shape[0], -1)
                # (batch, channels, H, W) → (batch, channels*H*W)
            outputs = layer(outputs)
        
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        first_linear = next((l for l in self.layers if isinstance(l, Linear)), None)
        
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
            
            if first_linear is not None and layer is first_linear and hasattr(self, 'flatten_input_shape'):
                grads = grads.reshape(self.flatten_input_shape)
        
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.conv_configs = param_list[0]   
        self.fc_configs   = param_list[1]   
        self.act_func     = param_list[2]   

        self.__init__(self.conv_configs, self.fc_configs, self.act_func)

        param_index = 3   
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[param_index]['W']
                layer.b = param_list[param_index]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[param_index]['weight_decay']
                layer.weight_decay_lambda = param_list[param_index]['lambda']
                param_index += 1
        
    def save_model(self, save_path):
        param_list = [self.conv_configs, self.fc_configs, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)