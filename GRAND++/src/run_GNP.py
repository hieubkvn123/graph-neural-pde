import sys, traceback
import argparse
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from GNP import GNP as GNN
from GNP_early import GNPEarly as GNNEarly
import time
from data import get_dataset, set_train_val_test_split
from ogb.nodeproppred import Evaluator
from best_params import best_params_dict
from recorder import Recorder
import os
import random
import time
import fcntl


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat)
    if model.opt['dataset'] == 'ogbn-arxiv':
        lf = torch.nn.functional.nll_loss
        loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        regularization_coeffs = model.regularization_coeffs

        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )
        loss = loss + reg_loss

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()
    return loss.item()


@torch.no_grad()
def test(model, data, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    # import pdb; pdb.set_trace()
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, opt):
    if opt['dataset'] == 'ogbn-arxiv':
        name = 'ogbn-arxiv'

    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

    evaluator = Evaluator(name=name)
    model.eval()

    out = model(feat).log_softmax(dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']

    return train_acc, valid_acc, test_acc

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

def main(cmd_opt):
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    opt['block'] = cmd_opt['block']
    opt['function'] = cmd_opt['function']
    opt['time'] = cmd_opt['time']
    opt['max_nfe'] = cmd_opt['max_nfe']
    opt['epoch'] = cmd_opt['epoch']

    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])
    np.random.RandomState(opt['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt['seed'])

    dataset = get_dataset(opt, '../data', opt['not_lcc'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # import pdb; pdb.set_trace()
    if opt['trusted_mask']:
        mask = dataset.data.train_mask.to(device)
    else:
        mask = None
    model = GNN(opt, dataset, device, trusted_mask=mask).to(device) if opt["no_early"] \
        else GNNEarly(opt, dataset, device, trusted_mask=mask).to(device)
    print(opt)
    if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset.data = set_train_val_test_split(opt['seed'], dataset.data,
                                                num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500,
                                                num_per_class=opt['num_train_per_class'])
    # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.

    data = dataset.data.to(device)
    noise = torch.randn_like(data.x) * opt['noise']
    if opt['noise_pos'] == 'test':
        noise *= (~data.train_mask)[:, None]
    data.x += noise
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = val_acc = test_acc = train_acc = best_epoch = 0
    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
    for epoch in range(1, opt['epoch']):
        start_time = time.time()

        tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)
        loss = train(model, optimizer, data)

        if tmp_val_acc > val_acc:
            best_epoch = epoch
            train_acc = tmp_train_acc
            val_acc = tmp_val_acc
            test_acc = tmp_test_acc
            best_time = opt['time']
        if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
            best_epoch = epoch
            val_acc = model.odeblock.test_integrator.solver.best_val
            test_acc = model.odeblock.test_integrator.solver.best_test
            train_acc = model.odeblock.test_integrator.solver.best_train
            best_time = model.odeblock.test_integrator.solver.best_time

        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

        print(
            log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc,
                       best_time))
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc,
                                                                                                         test_acc,
                                                                                                         best_epoch,
                                                                                                         best_time))
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trusted_mask', action='store_true')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--noise_pos', type=str, help='all, test')
    parser.add_argument('--prediffuse', action='store_true')
    parser.add_argument('--x0', action='store_true')
    parser.add_argument('--nox0', action='store_true')
    parser.add_argument('--icxb', type=float, default=1.0)
    parser.add_argument('--source_scale', type=float, default=1.0)
    parser.add_argument('--alltime', action='store_true')
    parser.add_argument('--allnumtrain', action='store_true')

    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # DeepGRAND args
    parser.add_argument("--alpha_", type=float, required=False, default=1.0, help='Alpha value - DeepGRAND')
    parser.add_argument("--epsilon_", type=float, required=False, default=1e-6, help='Epsilon value - DeepGRAND')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=100000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

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
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
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
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")

    parser.add_argument("--num_train_per_class", type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='../ray_tune', help="where to save results")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')

    args = parser.parse_args()

    opt = vars(args)

    start_time = time.time()
    rec = Recorder()
    mean_rec = Recorder()
    dirname = os.path.dirname(__file__)
    if opt['alltime']:
        time_list = [1.0, 4.0, 16.0, 32.0]#, 64.0, 128.0, 256.0]  # [1.0, 2.0, 4.0, 8.0, 16.0, 18.3, 32.0, 64.0, 128.0, 256.0]
        name = 'GNP_{}_{}.csv'.format(opt['block'][0], opt['dataset'])
    else:
        time_list = [opt['time']]
        name = 'GNP_{}_{}_{}.csv'.format(opt['block'][0], opt['dataset'], opt['time'])

    if opt['allnumtrain']:
        ntpc_list = [20, 10, 5, 2, 1]
    else:
        ntpc_list = [opt['num_train_per_class']]


    for ntpc in ntpc_list:
        opt['num_train_per_class'] = ntpc
        for x0 in [True, False]:
            if opt['nox0'] and x0 is True:
                continue
            opt['x0'] = x0
            for i in range(opt['num_runs']):
                run_start_time = time.time()
                for t in time_list:
                    opt['time'] = t
                    try:
                        print('time {} run {}'.format(t, i))
                        opt['seed'] = i
                        np.random.seed(opt['seed'])
                        torch.manual_seed(opt['seed'])
                        random.seed(opt['seed'])
                        np.random.RandomState(opt['seed'])
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(opt['seed'])

                        train_acc_val, val_acc_val, test_acc_val = main(opt)

                        t_rep = str(int(t)).zfill(3)
                        rec[t_rep] = test_acc_val
                        mean_rec[t_rep] = test_acc_val
                    except Exception:
                        traceback.print_exc(file=sys.stdout)
                        print("Exception occur, probably MAX NFE is achieved")
                rec['#time_elapsed'] = (time.time() - run_start_time) / 3600.0
                rec['#x0'] = int(x0)
                rec['#ntpc'] = ntpc
                rec['#numruns'] = opt['num_runs']
                rec['#runnum'] = i
                rec.capture(verbose=True)
                rec.writecsv(
                    os.path.join(dirname, '../output/' + name))

            mean_rec['#x0'] = int(x0)
            mean_rec['#ntpc'] = ntpc
            mean_rec['#numruns'] = opt['num_runs']
            mean_rec.capture(verbose=True)
            mean_rec.writecsv(
                os.path.join(dirname, '../output/mean_'+name))

    time_elapsed = (time.time() - start_time) / 3600.0
    print('time elapsed', time_elapsed, 'hours')

    """
    with open("/tanData/graph_poisson_network/results/all_results_gnp_new.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("\n\n%s\n" % opt["exp_name"])
        f.write('all train acc {}\n'.format(train_acc_list))
        f.write('all val acc {}\n'.format(val_acc_list))
        f.write('all test acc {}\n'.format(test_acc_list))
        f.write(
            'mean results: mean best val accuracy {:03f} with mean test accuracy {:03f} and mean train accuracy {:03f} after {} runs'.format(
                val_acc, test_acc, train_acc, opt['num_runs']))
        fcntl.flock(f, fcntl.LOCK_UN)
    """
