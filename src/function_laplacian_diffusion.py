import torch
import numpy as np
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
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
    f = alpha* (ax-x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f

class ExtendedLaplacianODEFunc(ODEFunc):
  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device, alpha_=1.0):
    super(ExtendedLaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.alpha_ = alpha_

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

    # Shape = 2045 x 80 (2045 = Number of nodes; 80 = Feature shape)
    ax = self.sparse_multiply(x)

    # Shape = (2045, ) (norm along dim 1)
    x_norm = torch.linalg.norm(x, 2, dim=1)

    # Shape = (2045, 1)
    x_norm = x_norm.view(-1, 1)

    f = (ax - x) * (x_norm ** self.alpha_) * 1e-6

    # Check if norm of f explodes 
    # norm_f = torch.linalg.norm(f, 1, dim=1)
    # norm_f = torch.mean(norm_f)
    # print("Mean of first order norm |f(X)| = ", norm_f.item())

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    return f


class ExtendedLaplacianODEFunc2(ODEFunc):
  # Set global attributes
  alpha_ = 1.0
  clipping_bound = 0.05

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(ExtendedLaplacianODEFunc2, self).__init__(opt, data, device)
    
    ### Log information ###
    print('****************** Extended Laplacian Function V.2 ******************')
    print('Clipping Bound = ', self.clipping_bound)
    print('Alpha = ', self.alpha_)
    print('*********************************************************************')

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
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
    
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    # Shape = 2045 x 80 (2045 = Number of nodes; 80 = Feature shape)
    ax = self.sparse_multiply(x)

    # Shape = (2045, ) (norm along dim 1)
    x_norm = torch.linalg.norm(x, 2, dim=1)
    
    # Truncate x_norm the have max=1
    x_norm = torch.clamp(x_norm, min=None, max=self.clipping_bound)

    # Shape = (2045, 1)
    x_norm = x_norm.view(-1, 1)

    # Previously : f = (ax - x) * (x_norm ** self.alpha_) * 1e-6
    f = (ax * (x_norm ** self.alpha_) - x) # * 1e-6

    # Check if norm of f explodes 
    # norm_f = torch.linalg.norm(f, 1, dim=1)
    # norm_f = torch.mean(norm_f)
    # print("Mean of first order norm |f(X)| = ", norm_f.item())

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    return f


class ExtendedLaplacianODEFunc3(ODEFunc):
  # Set global attributes
  alpha_ = 1.0
  clipping_bound = 0.05
  k = 5

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(ExtendedLaplacianODEFunc3, self).__init__(opt, data, device)

    ### Log information ###
    print('****************** Extended Laplacian Function V.3 ******************')
    print('K = ', self.k)
    print('Alpha = ', self.alpha_)
    print('*********************************************************************')

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

    # For attention matrices
    self.A = None
    self.P_inv = None

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def construct_dense_att_matrix(self, edge_index, edge_weight, num_node):
    att_matrix = torch.zeros((num_node, num_node))

    for i, (src, dst) in enumerate(zip(edge_index[0,:], edge_index[1,:])):
        att_matrix[src, dst] = edge_weight[i]

    return att_matrix

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    # Shape = 2045 x 80 (2045 = Number of nodes; 80 = Feature shape)
    ax = self.sparse_multiply(x) # Original AX
    I = torch.eye(x.shape[0]).to(x)

    # Construct negative-semidefinite A
    if(self.A is None):
        if(self.opt['block'] == 'attention'):
            self.A = self.construct_dense_att_matrix(self.edge_index, self.attention_weights.mean(dim=1), x.shape[0]).to(x)
        elif(self.opt['block'] == 'constant'):
            self.A = self.construct_dense_att_matrix(self.edge_index, self.edge_weight, x.shape[0]).to(x)

        # self.A = torch.transpose(self.A, 0, 1) # Make A right-stochastic
        self.A = (self.A - self.A.T)/2

    # Calculate Column-wise norm of X
    colwise_norm = torch.norm(x, p=2, dim=0)

    # Compute AX
    # ax = torch.matmul(self.A - I * (1 + 10 ** (-self.k)), x)

    # Formula (18)
    ax = torch.matmul(self.A, x)
    f = ax # * (colwise_norm ** self.alpha_)
    
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    return f

class NormLaplacianODEFunc(ODEFunc):
  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(NormLaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
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

    x_norm = torch.sqrt(torch.linalg.norm(x, 2, dim=1))
    x_norm = x_norm.view(-1, 1)
    x = x/x_norm

    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha* (ax-x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
