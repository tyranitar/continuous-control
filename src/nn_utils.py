from torch import nn
import math

def init_orth(layer, w_scale=1):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)

    return layer

def init_uniform(layer, lim):
    layer.weight.data.uniform_(-lim, lim)

    return layer

def init_sqrt_fan_in(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(fan_in)

    return init_uniform(layer, lim)
