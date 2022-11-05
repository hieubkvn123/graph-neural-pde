"""
A GNN used at test time that supports early stopping during the integrator
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import poi
import torchdiffeq

from base_classes import BaseGNN
from data import get_dataset
# from run_GNN import get_optimizer, train, test
from early_stop_solver import EarlyStopInt, EarlyStopIntPoi
from model_configurations import set_block, set_function


class GNPEarly(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cpu'), trusted_mask=None):
        super(GNPEarly, self).__init__(opt, dataset, device)
        self.trusted_mask = trusted_mask if opt['trusted_mask'] else None
        self.f = set_function(opt)
        self.m3 = nn.Linear(self.num_features, opt['hidden_dim'], bias=False)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        pre_tensor = torch.tensor([0, 1.0]).to(device)
        # self.regularization_fns = ()
        print("time", self.T)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
        self.preblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=pre_tensor).to(device)
        # overwrite the test integrator with this custom one
        if opt['adjoint']:
            self.odeblock.train_integrator = poi.poiint_adjoint
            self.odeblock.test_integrator = EarlyStopIntPoi(self.T, self.opt, self.device)
        else:
            self.odeblock.train_integrator = poi.poiint
            self.odeblock.test_integrator = EarlyStopIntPoi(self.T, self.opt, self.device)

        self.set_solver_data(dataset.data)  # tan: check this

    def set_solver_m2(self):
        if self.odeblock.test_integrator.m2 is None:
            self.odeblock.test_integrator.m2 = self.m2
            self.preblock.test_integrator.m2 = self.m2
        else:
            self.odeblock.test_integrator.m2.weight.data = self.m2.weight.data
            self.odeblock.test_integrator.m2.bias.data = self.m2.bias.data
            self.preblock.test_integrator.m2.weight.data = self.m2.weight.data
            self.preblock.test_integrator.m2.bias.data = self.m2.bias.data

    def set_solver_data(self, data):
        self.odeblock.test_integrator.data = data

    def forward(self, x):
        # Encode each node based on its feature.
        if self.opt['use_labels']:
            y = x[:, self.num_features:]
            x = x[:, :self.num_features]
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        if self.opt['use_labels']:
            x = torch.cat([x, y], dim=-1)

        if self.opt['batch_norm']:
            x = self.bn_in(x)

        # Solve the initial value problem of the ODE.
        if self.opt['augment']:
            c_aux = torch.zeros(x.shape).to(self.device)
            x = torch.cat([x, c_aux], dim=1)

        self.odeblock.set_x0(x * self.opt['x0'])
        self.preblock.set_x0(x * self.opt['x0'])
        self.set_solver_m2()

        if self.trusted_mask is not None:
            xp = x * self.trusted_mask[:, None]
            ave = xp.sum(dim=0) / self.trusted_mask.sum(dim=0)
            xp -= ave[None, :]
            xp *= self.trusted_mask[:, None]
            xb = xp
        else:
            xb = x

        if self.opt['prediffuse']:
            xb = self.preblock(xb)

        xb = xb * self.opt['source_scale']

        z = self.odeblock(x + xb * self.opt['icxb'], icc=1, c0=xb)

        if self.opt['augment']:
            z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        # Activation.
        z = F.relu(z)

        if self.opt['fc_out']:
            z = self.fc(z)
            z = F.relu(z)

        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z


def main(opt):
    dataset = get_dataset(opt, '../data', False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = GNPEarly(opt, dataset, device).to(device), dataset.data.to(device)
    print(opt)
    # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_val_acc = test_acc = best_epoch = 0
    best_val_acc_int = best_test_acc_int = best_epoch_int = 0
    for epoch in range(1, opt['epoch']):
        start_time = time.time()
        loss = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        val_acc_int = model.odeblock.test_integrator.solver.best_val
        tmp_test_acc_int = model.odeblock.test_integrator.solver.best_test
        # store best stuff inside integrator forward pass
        if val_acc_int > best_val_acc_int:
            best_val_acc_int = val_acc_int
            test_acc_int = tmp_test_acc_int
            best_epoch_int = epoch
        # store best stuff at the end of integrator forward pass
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_epoch = epoch
        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(
            log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc,
                       tmp_test_acc))
        log = 'Performance inside integrator Val: {:.4f}, Test: {:.4f}'
        print(log.format(val_acc_int, tmp_test_acc_int))
        # print(
        # log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))
    print('best in integrator val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc_int,
                                                                                                  test_acc_int,
                                                                                                  best_epoch_int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    # ODE args
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument(
        "--adjoint_method", type=str, default="adaptive_heun",
        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
    parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

    args = parser.parse_args()

    opt = vars(args)

    main(opt)
