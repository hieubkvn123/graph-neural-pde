#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import torch
from torch import tensor
from torch import nn
from data import get_dataset
from function_laplacian_diffusion import LaplacianODEFunc
from GNN import GNN
from block_transformer_attention import AttODEblock


class AttentionODEBlockTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.opt = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
                'attention_norm_idx': 0, 'simple': True, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'block': 'attention', 'function': 'laplacian', 'alpha_sigmoid': True, 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler'}
    self.dataset = get_dataset('Cora', '../data', False)

  def tearDown(self) -> None:
    pass

  def test_block(self):
    data = self.dataset.data
    self.opt['hidden_dim'] = self.dataset.num_features
    self.opt['heads'] = 1
    gnn = GNN(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, AttODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    out = odeblock(data.x)
    print('ode block out', out)
    self.assertTrue(data.x.shape == out.shape)
    self.opt['heads'] = 2
    try:
      gnn = GNN(self.opt, self.dataset, device=self.device)
      self.assertTrue(False)
    except AssertionError:
      pass

  def test_gnn(self):
    gnn = GNN(self.opt, self.dataset, device=self.device)
    gnn.train()
    out = gnn(self.dataset.data.x)
    print(out.shape)
    print(torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))
    self.assertTrue(out.shape == torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))
    gnn.eval()
    out = gnn(self.dataset.data.x)
    self.assertTrue(out.shape == torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))


if __name__ == '__main__':
  est = AttentionODEBlockTests()
  est.setUp()
  est.test_block()
