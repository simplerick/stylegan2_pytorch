import torch
import torch.nn as nn
import numpy as np
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



class Equal_LR:
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



def grid(array, ncols=8):
    """
    Makes grid from batch of images with shape (n_batch, height, width, channels)
    """
    array = np.pad(array, [(0,0),(1,1),(1,1),(0,0)], 'constant')
    nindex, height, width, intensity = array.shape
    nrows = (nindex+ncols-1)//ncols
    r = nrows*ncols - nindex # remainder
    # want result.shape = (height*nrows, width*ncols, intensity)
    arr = np.concatenate([array]+[np.zeros([1,height,width,intensity])]*r)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return np.pad(result, [(1,1),(1,1),(0,0)], 'constant')



class NextDataLoader(torch.utils.data.DataLoader):
    '''
    Dataloader with __next__ method
    '''
    def __next__(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = self.__iter__()
            return next(self.iterator)
