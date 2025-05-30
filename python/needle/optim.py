"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(lambda: 0.0)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # for param in self.params:
            
        #     if param.grad is None:
        #         continue
        #     grad  = self.u.get(param, 0) * self.momentum + (1-self.momentum) * (param.grad.data + self.weight_decay * param.data)
        #     grad = ndl.Tensor(grad, dtype = param.dtype)
        #     self.u[param] = grad
        #     param.data =  param.data - self.lr * grad
        for param in self.params:
            if param.grad is None:
                continue
            grad  = self.u.get(param, 0) * self.momentum + (1-self.momentum) * (param.grad.data + self.weight_decay * param.data)
            grad = ndl.Tensor(grad, dtype = param.dtype)
            self.u[param] = grad
            param.data =  param.data - self.lr * grad
        
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay * param.detach()
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad ** 2)
            m_t1_hat = self.m[param] / (1 - self.beta1 ** (self.t))
            v_t1_hat = self.v[param] / (1 - self.beta2 ** (self.t))
            param.cached_data -= (self.lr * m_t1_hat / ((v_t1_hat ** 0.5) + self.eps)).cached_data
                
        
