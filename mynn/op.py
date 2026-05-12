from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output=X@self.W+self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        grad_input = grad @ self.W.T
        return grad_input
        
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights: [out_channels, in_channels, kernel_size, kernel_size]
        fan_in = in_channels * kernel_size * kernel_size
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels,))
        
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process
        
        self.params = {'W': self.W, 'b': self.b}
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, _, H, W = X.shape
        out_channels, _, k, _ = self.W.shape
        
        # Calculate output dimensions
        H_out = (H - k) // self.stride + 1
        W_out = (W - k) // self.stride + 1
        
        # Use im2col + GEMM for faster convolution
        # im2col: convert input patches to columns
        col = self.im2col(X, k, self.stride)
        # Reshape weights to [out_channels, in_channels*k*k]
        W_reshaped = self.W.reshape(out_channels, -1)
        # Matrix multiplication: [out_channels, in_channels*k*k] @ [in_channels*k*k, batch*H_out*W_out]
        out = W_reshaped @ col
        # Reshape to output shape and add bias
        output = out.reshape(out_channels, batch_size, H_out, W_out).transpose(1, 0, 2, 3)
        output += self.b.reshape(1, -1, 1, 1)
        
        return output
    
    def im2col(self, X, kernel_size, stride):
        """Convert input to column matrix for efficient convolution"""
        batch_size, in_channels, H, W = X.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        
        col = np.zeros((in_channels * kernel_size * kernel_size, batch_size * H_out * W_out))
        
        col_idx = 0
        for b in range(batch_size):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    patch = X[b, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    col[:, col_idx] = patch.reshape(-1)
                    col_idx += 1
        
        return col

    def col2im(self, col, input_shape, kernel_size, stride):
        """Convert column matrix back to input shape for gradient computation"""
        batch_size, in_channels, H, W = input_shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        
        grad_input = np.zeros(input_shape)
        
        col_idx = 0
        for b in range(batch_size):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    patch_grad = col[:, col_idx].reshape(in_channels, kernel_size, kernel_size)
                    grad_input[b, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size] += patch_grad
                    col_idx += 1
        
        return grad_input

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, out_channels, H_out, W_out = grads.shape
        _, in_channels, k, _ = self.W.shape
        
        # Initialize gradients
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        
        # Initialize gradient for input
        grad_input = np.zeros_like(self.input)
        
        # Reshape grads to [out_channels, batch_size * H_out * W_out]
        grads_reshaped = grads.transpose(1, 0, 2, 3).reshape(out_channels, -1)
        
        # im2col for input
        col = self.im2col(self.input, k, self.stride)
        
        # Weight gradients: grads_reshaped @ col.T
        self.grads['W'] = (grads_reshaped @ col.T).reshape(self.W.shape)
        
        # Bias gradients: sum over batch and spatial dimensions
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3))
        
        # Input gradients: W.T @ grads_reshaped, then col2im
        W_reshaped = self.W.reshape(out_channels, -1)
        grad_col = W_reshaped.T @ grads_reshaped
        grad_input = self.col2im(grad_col, self.input.shape, k, self.stride)
        
        # Apply weight decay
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.optimizable = False
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.input = None  # To store the predictions for backward

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.input = predicts
        self.labels = labels  
        batch_size = predicts.shape[0]
        
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        
        loss = -np.sum(np.log(probs[np.arange(batch_size), labels]+1e-7)) / batch_size
        
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.input.shape[0]
        
        if self.has_softmax:
            probs = softmax(self.input)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch_size), self.labels] = 1
            self.grads = (probs - one_hot) / batch_size

        else:
            probs = self.input
            self.grads = np.zeros_like(probs)
            self.grads[np.arange(batch_size), self.labels] = \
                -1.0 / (probs[np.arange(batch_size), self.labels] * batch_size)
            
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, lambda_=1e-4):
        super().__init__()
        self.optimizable = False
        self.lambda_ = lambda_
        self.input = None

    def __call__(self, W):
        return self.forward(W)

    def forward(self, W):
        self.input = W
        return self.lambda_ * np.sum(W**2)

    def backward(self):
        return 2 * self.lambda_ * self.input
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition