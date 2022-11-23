import torch
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function

class EquivGNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(EquivGNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    self.device = device
    time_tensor = torch.tensor([0, self.T]).to(device)
    # self.regularization_fns = ()
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    # overwrite the test integrator with this custom one
    # with torch.no_grad():
    #   self.odeblock.test_integrator = EarlyStopInt(self.T, self.opt, self.device)
    #   self.set_solver_data(dataset.data)

  # def set_solver_m2(self):
  #   self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
  #   self.odeblock.test_integrator.m2_bias = self.m2.bias.data.detach().clone().to(self.device)

  # def set_solver_data(self, data):
  #   self.odeblock.test_integrator.data = data


  # def cleanup(self):
  #   del self.odeblock.test_integrator.m2
  #   torch.cuda.empty_cache()


  def forward(self, h, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['beltrami']:
      h = F.dropout(h, self.opt['input_dropout'], training=self.training)
      h = self.mx(h)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      h = torch.cat([h, p], dim=1)
    else:
      h = F.dropout(h, self.opt['input_dropout'], training=self.training)
      h = self.m1(h)

    if self.opt['use_mlp']:
      h = F.dropout(h, self.opt['dropout'], training=self.training)
      h = F.dropout(h + self.m11(F.relu(h)), self.opt['dropout'], training=self.training)
      h = F.dropout(h + self.m12(F.relu(h)), self.opt['dropout'], training=self.training)

    if self.opt['batch_norm']:
      h = self.bn_in(h)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(h.shape).to(self.device)
      h = torch.cat([h, c_aux], dim=1)

    self.odeblock.set_x0(h)

    # with torch.no_grad():
    #   self.set_solver_m2()

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states  = self.odeblock((h, x))
    else:
      z = self.odeblock((h, x))
      
    if self.opt['augment']:
      z = torch.split(z, h.shape[1] // 2, dim=1)[0]

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