import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    a = math.sqrt(6 / (fan_in + fan_out)) * gain
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    std = math.sqrt(2 / (fan_in + fan_out)) * gain
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    a = math.sqrt(2) * math.sqrt(3/fan_in)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION