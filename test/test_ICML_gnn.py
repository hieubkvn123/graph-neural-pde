#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import unittest
import torch
from torch import tensor
from torch import nn
from GNN_ICML import gcn_norm_fill_val, coo2tensor, train_ray
from data import get_dataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from ray.tune.utils import diagnose_serialization
from functools import partial
import os


class ICMLGNNTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0,2,2],[1,0,1]])
    self.x = tensor([[1.,2.],[3.,2.],[4.,5.]], dtype=float)
    self.W = tensor([[2,1],[3,2]], dtype=float)
    self.alpha = tensor([[1,2,3,4]], dtype=float)
    self.leakyrelu = nn.LeakyReLU(0.2)

  def tearDown(self) -> None:
    pass

  def test_fill_norm(self):
    opt = {'dataset': 'Cora', 'improved': False, 'self_loop_weight': 1.}
    dataset = get_dataset(opt['dataset'], '../data', False)
    data = dataset.data
    edge_index1, edge_weight1 = gcn_norm(data.edge_index, data.edge_attr, data.num_nodes,
                                         opt['improved'], opt['self_loop_weight'] > 0, dtype=data.x.dtype)
    edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, data.edge_attr, opt['self_loop_weight'],
                                                data.num_nodes, dtype=data.x.dtype)
    assert torch.all(edge_index.eq(edge_index1))
    assert torch.all(edge_weight.eq(edge_weight1))


def main():
  data_dir = os.path.abspath("../data")
  trainable = partial(train_ray, data_dir=data_dir)
  diagnose_serialization(trainable)
  opt = {'dataset': 'Cora', 'improved': False, 'self_loop_weight': 1.}
  dataset = get_dataset(opt['dataset'], '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = dataset.data
  edge_index1, edge_weight1 = gcn_norm(data.edge_index, data.edge_attr, data.num_nodes,
    opt['improved'], opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, data.edge_attr, opt['self_loop_weight'], data.num_nodes,
                                              opt['self_loop_weight'] > 0)
  assert torch.all(edge_index.eq(edge_index1))
  assert torch.all(edge_weight.eq(edge_weight1))
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  coo = coo2tensor(coo, device)


if __name__ == '__main__':
  main()