import torch
import torchdiffeq
from torch import nn


def split(tensor, dim, splits):
    assert tensor.dim() >= dim, "Can't find dimension {} in tensor of dimensions {}".format(dim, tensor.dim())
    assert tensor.shape[dim] % splits == 0, "Length {} on dimension {} does not divide by splits {}" \
        .format(tensor.shape[dim], dim, splits)
    split_size = tensor.shape[dim] // splits
    out = [torch.narrow(tensor, dim, i * split_size, split_size) for i in range(splits)]
    return out


class WrapConst(nn.Module):
    def __init__(self, func, dim):
        super(WrapConst, self).__init__()
        self.func = func
        self.dim = dim

    def forward(self, t, z, *args, **kwargs):
        # dz0 = func(z0) + z1, dz1 = 0
        z0, z1 = split(z, self.dim, 2)
        dz0 = self.func(t, z0, *args, **kwargs) + z1
        dz1 = torch.zeros_like(z1)
        dz = torch.cat([dz0, dz1], dim=self.dim)
        return dz


def PoissonIntegration(odeint, func, y0, t, *args, icc=0, stab_dim=None, c0=None, wrap_dim=0, **kwargs):
    c0 = y0 if c0 is None else c0
    if stab_dim is not None:
        c0 = c0 - torch.mean(c0, dim=stab_dim, keepdim=True)
    ic = torch.cat([y0 * icc, c0], dim=wrap_dim)
    wrap_func = WrapConst(func, wrap_dim)
    odeout = odeint(wrap_func, ic, t, *args, **kwargs)
    out = split(odeout, wrap_dim + 1, 2)[0]
    return out


def poiint(func, y0, t, *args, **kwargs):
    return PoissonIntegration(torchdiffeq.odeint, func, y0, t, *args, **kwargs)


def poiint_adjoint(func, y0, t, *args, **kwargs):
    return PoissonIntegration(torchdiffeq.odeint_adjoint, func, y0, t, *args, **kwargs)