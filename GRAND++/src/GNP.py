import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function



# Define the GNN model.
class GNP(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cpu'), trusted_mask=None):
        super(GNP, self).__init__(opt, dataset, device)
        self.trusted_mask = trusted_mask
        self.f = set_function(opt)
        self.m3 = nn.Linear(self.num_features, opt['hidden_dim'], bias=False)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

    def forward(self, x):
        # Encode each node based on its feature.
        if self.opt['use_labels']:
            y = x[:, self.num_features:]
            x = x[:, :self.num_features]
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        if self.opt['use_mlp']:
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
        # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

        if self.opt['use_labels']:
            x = torch.cat([x, y], dim=-1)

        if self.opt['batch_norm']:
            x = self.bn_in(x)

        # Solve the initial value problem of the ODE.
        if self.opt['augment']:
            c_aux = torch.zeros(x.shape).to(self.device)
            x = torch.cat([x, c_aux], dim=1)

        self.odeblock.set_x0(x * 0)

        xp = x
        if self.trusted_mask is not None:
            xp = xp * self.trusted_mask[:, None]
            ave = xp.sum(dim=0) / self.trusted_mask.sum(dim=0)
            xp -= ave[None, :]
            xp *= self.trusted_mask
        z = self.odeblock(xp)  # TODO: c0 {None, xp}

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
