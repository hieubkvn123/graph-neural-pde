import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException


class LaplacianODEFunc(ODEFunc):
  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)
    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_train = nn.Parameter(torch.ones(1), requires_grad=False)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax-x) 
    return f

class ExtendedLaplacianODEFunc3(ODEFunc):
  # Set global attributes
  alpha_ = 1.0
  epsilon_ = 1e-6
  clipping_bound = 0.05

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(ExtendedLaplacianODEFunc3, self).__init__(opt, data, device)

    ### Log information ###
    print('****************** Extended Laplacian Function V.3 ******************')
    print('Alpha = ', self.alpha_)
    print('Epsilon = ', self.epsilon_)
    print('*********************************************************************')

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.norm_scaler = nn.Parameter(0.1 * torch.ones(opt['hidden_dim']))
    self.epsilon_params = nn.Parameter(torch.Tensor([self.epsilon_]))
    self.alpha_params = nn.Parameter(torch.Tensor([self.alpha_]))

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    ax = self.sparse_multiply(x)
    x_norm = torch.linalg.norm(x, 2, dim=0)

    if(self.opt['alpha_learnable']):
        f = alpha * (ax - (1 + self.epsilon_params) * x) * (x_norm ** self.alpha_params) # With learnable alpha-epsilon 
    else:
        f = alpha * (ax - (1 + self.epsilon_) * x) * (x_norm ** self.alpha_) 

    return f
