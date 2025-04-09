"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features,
                                           device=device, dtype=dtype))
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = self.bias.transpose()
            self.bias = Parameter(self.bias, device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        m = X.shape[0]
        return X @ self.weight + ops.broadcast_to(self.bias, (m, self.out_features))
        # if self.bias.shape != (1, self.out_features):
        #     self.bias = self.bias.reshape((1, self.out_features))
        # y = ops.matmul(X, self.weight)
        # if self.bias:
        #     y += self.bias.broadcast_to(y.shape)
        # return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, label_size = logits.shape
        one_hot_y = init.one_hot(label_size, y)
        true_logits = ops.summation(logits * one_hot_y, axes=(1,))
        return (ops.logsumexp(logits, axes=(1, )) - true_logits).sum()/batch_size
        ### END YOUR SOLUTION




class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1,dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(1,dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            b = x.shape[0]
            mean = (ops.summation(x, axes=(0,), keepdims=True) / b)
            var = (ops.summation(ops.power_scalar((x - mean.broadcast_to(x.shape)), 2), axes=(0,), keepdims=True) / b)
            self.running_mean = self.running_mean * (1 - self.momentum) + mean.reshape(self.running_mean.shape).detach() * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var.reshape(self.running_var.shape).detach() * self.momentum
            mean = mean.broadcast_to(x.shape)
            var = var.broadcast_to(x.shape)
            x_hat = (x - mean) / ops.power_scalar((var + self.eps), 0.5)
        else:
            x_hat = (x - self.running_mean.broadcast_to(x.shape)) / ops.power_scalar((self.running_var.broadcast_to(x.shape) + self.eps), 0.5)
        return self.weight.broadcast_to(x.shape) * x_hat + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.w = Parameter(init.ones(1,dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.b = Parameter(init.zeros(1,dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # mean: b,m,n
        mean = (ops.summation(x, axes=(-1,), keepdims=True) / self.dim).broadcast_to(x.shape)
        # var: b,m,n
        var = (ops.summation((x - mean)**2, axes=(-1,), keepdims=True) / self.dim).broadcast_to(x.shape)
        x_hat = (x - mean) / ops.power_scalar((var + self.eps), 0.5)
        return self.w.broadcast_to(x.shape) * x_hat + self.b.broadcast_to(x.shape)
        # m*n
        # mean = (x.sum((-1,)) /
        #         x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # var = (((x - mean)**2).sum((-1,)) /
        #        x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # deno = (var + self.eps)**0.5
        # return self.w.broadcast_to(x.shape) * (x - mean) / deno + self.b.broadcast_to(x.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device)
            x = x * mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION




class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = x + self.fn(x)
        return x
        ### END YOUR SOLUTION
