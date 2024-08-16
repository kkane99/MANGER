import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
import pdb

class NSEPRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NSEPRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.mlp = nn.ModuleList(
            [nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(args.n_agents)])

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, update_metric):
        b, a, e = inputs.size()
        # calculate Q_total
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        rnn_output = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(rnn_output)).reshape(b, a, -1)
        else:
            q = self.fc2(rnn_output).reshape(b, a, -1)
        # calculate Q_sep
        rnn_output_reshape = rnn_output.reshape(b,a,-1).detach()
        q_sep = th.stack([mlp(rnn_output_reshape[:,id, :])for id, mlp in enumerate(self.mlp)], dim=1)
        if update_metric is None:
            q_sum = q + q_sep*0.5
        else:
            update_metric = update_metric.bool().float().unsqueeze(-1).repeat(1,1,q.shape[-1])
            # q_sum = q*update_metric + q*(1-update_metric).detach() + q_sep*0.1
            q_sum = q.detach() + 0.5*(q_sep*update_metric + q_sep*(1-update_metric).detach())
        return q_sum, rnn_output_reshape