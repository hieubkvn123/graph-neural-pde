from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val


class ConstantPoiODEblock(ODEblock):
    def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
        super(ConstantPoiODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

        self.aug_dim = 2 if opt['augment'] else 1
        self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
        if opt['data_norm'] == 'rw':
            edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                 fill_value=opt['self_loop_weight'],
                                                 num_nodes=data.num_nodes,
                                                 dtype=data.x.dtype)
        else:
            edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                                        fill_value=opt['self_loop_weight'],
                                                        num_nodes=data.num_nodes,
                                                        dtype=data.x.dtype)
        self.odefunc.edge_index = edge_index.to(device)
        self.odefunc.edge_weight = edge_weight.to(device)
        self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

        if opt['adjoint']:
            from poi import poiint_adjoint as poiint
        else:
            from poi import poiint

        self.train_integrator = poiint
        self.test_integrator = poiint
        self.set_tol()

    def forward(self, x, c0=None, **kwargs):
        t = self.t.type_as(x)

        integrator = self.train_integrator if self.training else self.test_integrator

        reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))

        func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
        state = (x,) + reg_states if self.training and self.nreg > 0 else x

        if self.opt["adjoint"] and self.training:
            state_dt = integrator(
                func, state, t,
                c0=c0,
                method=self.opt['method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(step_size=self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint, **kwargs)
        else:
            state_dt = integrator(
                func, state, t,
                c0=c0,
                method=self.opt['method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol, **kwargs)

        if self.training and self.nreg > 0:
            z = state_dt[0][1]
            reg_states = tuple(st[1] for st in state_dt[1:])
            return z, reg_states
        else:
            z = state_dt[1]
            return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"
