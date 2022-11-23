import torch
from torch import nn
import torch_sparse
from torch_geometric.utils import softmax
import numpy as np 
from base_classes import ODEFunc
from utils import MaxNFEException, squareplus

class EquivTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """
  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(EquivTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h
      
    self.Q = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.Q)
    
    # create for the calculation of coordinate difference
    self.Q_rad = nn.Linear(1, self.attention_dim)
    self.init_weights(self.Q_rad)
    
    self.K = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.K)
      
    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, in_features, edge):
    """
    in_features might be [features, radial]
    """
    hi, radial = in_features
    
    q_rad = self.Q_rad(radial)
    q = self.Q(hi)
    k = self.K(hi)

    # perform linear operation and split into h heads

    k = k.view(-1, self.h, self.d_k)
    q = q.view(-1, self.h, self.d_k)
    q_rad = q_rad.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_nodes, attention_dim, n_heads]

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    q_rad = q_rad.transpose(1, 2)

    src = q[edge[0, :], :, :] + q_rad 
    dst_k = k[edge[1, :], :, :]

    prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)

    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)
    if self.opt['square_plus']:
      attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
    else:
      attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    return attention, prods

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class EquivLaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(EquivLaplacianODEFunc, self).__init__(opt, data, device)

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

  def forward(self, t, in_features):  # the t param is needed by the ODE solver.
    h, x = in_features
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    ah = self.sparse_multiply(h)    
    ax = self.sparse_multiply(x)
    
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    fh = alpha * (ah - h)
    fx = alpha * (ax - x)
    
    # if self.opt['add_source']:
    #   fh = fh + self.beta_train * self.x0
      
    return (fh, fx)