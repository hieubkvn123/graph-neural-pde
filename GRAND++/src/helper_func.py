import numpy as np


def param_hash(d):
    """
    Hash a parameter dictionary with datatype {string: float}
    :param d: dictionary, value of type float
    :return: int, hash number
    """
    sum = 0
    for key, val in d.items():
        key_val = len(key) * ord(key[0]) * ord(key[-1])
        temp = np.sin(key_val) * np.cos(val)
        sum += int((temp + 1) * 1e10)
    return sum


def tensor_to_np(x):
    """
    Cast a tensor to numpy
    :param x: type torch.Tensor
    :return x: type np.array
    """
    for method in ['detach', 'cpu', 'numpy']:
        if hasattr(x, method):
            x = getattr(x, method)()
    return x


def count_parameters(model):
    """
    Compute the number of trainable parameters of the model.
    :param model: type nn.Module
    :return: number of parameters, type int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shrink_parameters(model, ratio):
    """
    Shrink all parameters of a model to a certain ratio
    :param model: type nn.Module
    :param ratio: type float
    :return:
    """
    model_dict = model.state_dict()
    for i in model_dict:
        model_dict[i] *= ratio
    model.load_state_dict(model_dict)
    return model
