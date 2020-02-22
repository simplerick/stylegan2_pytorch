import torch
import torch.nn as nn
from copy import deepcopy


class Scaled_Act(nn.Module):
    '''
    Scale nonlinearity. By default it scales to retain signal variance
    '''
    to_str = {'Sigmoid' : 'sigmoid', 'ReLU': 'relu', 'Tanh' : 'tanh', 'LeakyReLU': 'leaky_relu'}
    def __init__(self, act, scale = None):
        super().__init__()
        self.act = act
        act_name = Scaled_Act.to_str.get(act._get_name(), act._get_name())
        param = getattr(act, 'negative_slope', None)
        self.scale = scale if scale else torch.nn.init.calculate_gain(act_name, param)

    def forward(self, input):
        return self.scale*self.act(input)



class EqualLR:
    '''
    Equalized learning rate. Applies recursively to all submodules.
    '''
    def __init__(self, name):
        self.name = name

    @staticmethod
    def compute_norm(module, weight):
        mode = 'fan_in'
        if hasattr(module, 'transposed') and module.transposed:
            mode = 'fan_out'
        return torch.nn.init._calculate_correct_fan(weight, mode)

    @staticmethod
    def scale(module, input, output):
        return module.scale*output

    def fn(self, module):
        try:
            weight = getattr(module, self.name)
            module.scale = 1/np.sqrt(EqualLR.compute_norm(module, weight))
            module.equalize = module.register_forward_hook(EqualLR.scale)
        except:
            pass

    def __call__(self, module):
        new_module = deepcopy(module)
        new_module.apply(self.fn)
        return new_module
