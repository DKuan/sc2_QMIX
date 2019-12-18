# File name: model.py
# Author: Zachry
# Time: 2019-12-04
# Description: The model structure of QMIX
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super(Q_Network, self).__init__()
        self.hidden_size = args.q_net_hidden_size
        self.mlp_in_layer = nn.Linear(obs_size+act_size, args.q_net_out[0])
        self.mlp_out_layer = nn.Linear(args.q_net_hidden_size, act_size)
        self.GRU_layer = nn.GRUCell(args.q_net_out[0], args.q_net_hidden_size)
        self.ReLU = nn.ReLU()

        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_in_layer.weight)
        nn.init.xavier_uniform_(self.mlp_out_layer.weight)
    
    def init_hidden(self, args):
        return self.mlp_in_layer.weight.new(1, args.q_net_hidden_size).zero_()

    def forward(self, obs_a_cat, hidden_last):
        #x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        gru_out = self.GRU_layer(x, hidden_last)
        output = self.mlp_out_layer(gru_out)
        return output, gru_out
    
class Hyper_Network(nn.Module):
    def __init__(self, shape_state, shape_hyper_net, args):
        super(Hyper_Network, self).__init__()
        self.hyper_net_pars = shape_hyper_net
        self.w1_layer = nn.Linear(shape_state, shape_hyper_net['w1_size'])
        self.w2_layer = nn.Linear(shape_state, shape_hyper_net['w2_size'])
        self.b1_layer = nn.Linear(shape_state, shape_hyper_net['b1_size'])
        self.b2_layer_i = nn.Linear(shape_state, args.shape_hyper_b2_hidden)
        self.b2_layer_h = nn.Linear(args.shape_hyper_b2_hidden, shape_hyper_net['b2_size'])
        self.LReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()

        #self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer_i.weight)
        nn.init.xavier_uniform_(self.b2_layer_h.weight)

    def forward(self, state):
        w1_shape = self.hyper_net_pars['w1_shape']
        w2_shape = self.hyper_net_pars['w2_shape']
        w1 = torch.abs(self.w1_layer(state)).view(-1, w1_shape[0], w1_shape[1])
        w2 = torch.abs(self.w2_layer(state)).view(-1, w2_shape[0], w2_shape[1])
        b1 = self.b1_layer(state).view(-1, 1, self.hyper_net_pars['b1_shape'][0])
        #x = self.LReLU(self.b2_layer_i(state))
        x = self.ReLU(self.b2_layer_i(state))
        b2 = self.b2_layer_h(x).view(-1, 1, self.hyper_net_pars['b2_shape'][0])
        return {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
        
class Mixing_Network(nn.Module):
    def __init__(self, action_size, num_agents, args):
        super(Mixing_Network, self).__init__()
        # action_size * num_agents = the num of Q values
        self.w1_shape = torch.Size((num_agents, args.mix_net_out[0]))
        self.b1_shape = torch.Size((args.mix_net_out[0], ))
        self.w2_shape = torch.Size((args.mix_net_out[0], args.mix_net_out[1]))
        self.b2_shape = torch.Size((args.mix_net_out[1], ))
        self.w1_size = self.w1_shape[0] * self.w1_shape[1]
        self.b1_size = self.b1_shape[0]
        self.w2_size = self.w2_shape[0] * self.w2_shape[1]
        self.b2_size = self.b2_shape[0]
        self.pars = {'w1_shape':self.w1_shape, 'w1_size':self.w1_size, \
                'w2_shape':self.w2_shape, 'w2_size':self.w2_size, \
                'b1_shape':self.b1_shape, 'b1_size':self.b1_size, \
                'b2_shape':self.b2_shape, 'b2_size':self.b2_size, }
        self.LReLU = nn.LeakyReLU(0.001)
        self.ReLU = nn.ReLU()
    
    def forward(self, q_values, hyper_pars):
        x = self.ReLU(torch.bmm(q_values, hyper_pars['w1']) + hyper_pars['b1'])
        output = torch.bmm(x, hyper_pars['w2']) + hyper_pars['b2']
        return output.view(-1)
