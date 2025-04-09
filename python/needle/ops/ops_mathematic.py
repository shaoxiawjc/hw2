"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return (a + self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return (a * self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # y = a^b (y = a**1)
        # dy/da = b* a^(b-1)
        # dy/db = a^b * ln(b)
        a, b = node.inputs
        return multiply(out_grad, multiply(b, power(a, add_scalar(b, -1)))), multiply(out_grad, multiply(power(a,b), log(a)))
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.power(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # y = x^a
        # dy/dx = ax** (a-1)
        x = node.inputs[0]
        if self.scalar == 1:
            return multiply(out_grad, Tensor(array_api.ones(x.shape)))
        return multiply(out_grad, mul_scalar(power_scalar(x, self.scalar-1), self.scalar))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # print(len(a.shape) , len(b.shape))
        if len(a.shape) > 1 and len(b.shape) == 1:
            return array_api.true_divide(a, b.reshape(1, -1))
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # y = a/b
        # dy/da = 1/b
        # dy/db = a * -/b**2
        a,b = node.inputs[0], node.inputs[1]
        return multiply(out_grad, power_scalar(b, -1)), multiply(out_grad, negate(multiply(a, power_scalar(b, -2))))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.true_divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = node.inputs[0]
        # y = x/2

        if self.scalar == 1:
            return out_grad
        return mul_scalar(out_grad, 1/self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return array_api.swapaxes(a, x, y)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return transpose(out_grad, axes=(x, y))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.reshape(a, self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (reshape(out_grad, shape=x.shape), )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = node.inputs[0]
        # m*n*d
        # n*d
        # 广播只能在0维度之前进行
        ori_shape = x.shape
        now_shape = self.shape
        expand_axes = []
        for i in range(-1, -len(now_shape)-1, -1):
            if i < -len(ori_shape):
                expand_axes.append(i+len(now_shape))
                continue
            if now_shape[i] != ori_shape[i]:
                expand_axes.append(i+len(now_shape))
        dx = summation(out_grad, tuple(expand_axes), keepdims=False)
        dx = reshape(dx, shape=ori_shape)
        return dx
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims = False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # example:
        # a: 3,4,5
        # axis: 0,1
        # keepdims: True
        # result: 1,1,5
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # example: out_grad: 3,4
        # x: 3,4,5
        new_shape = list(node.inputs[0].shape)
        if self.axes == -1:
            axes = [-1]
        else:
            axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Y = A@B
        # A: m*n B: n*k # Y and dy: m*k
        # da = m*k*k*n
        # db = n*m*m*k
        # a, b = node.inputs
        # print(f"a shape: {a.shape}, transpose a shape: {transpose(a).shape}")
        # print(f"b shape: {b.shape}, transpose b shape: {transpose(b).shape}")
        # return matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        if len(grad_a.shape) != len(a.shape):
            grad_a = summation(grad_a, tuple(range(len(grad_a.shape) - len(a.shape))), keepdims=False)
        if len(grad_b.shape) != len(b.shape):
            grad_b = summation(grad_b, tuple(range(len(grad_b.shape) - len(b.shape))), keepdims=False)
        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.log(a )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = node.inputs[0]
        return multiply(out_grad, power_scalar(x, -1))
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # y = e^x
        # dy/dx = e^x
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = node.inputs[0]
        x_data = x.realize_cached_data()
        dx = Tensor((x_data > 0).astype(x.dtype))
        return out_grad * relu(dx)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

