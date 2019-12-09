# File name: model.py
# Author: Zachry
# Time: 2019-12-04
# Description: The model structure of QMIX
import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, obs_size, act_size, args):
        super(Q_Network, self).__init__()
        self.mlp_in_layer = nn.Linear(obs_size+act_size, args.q_net_out[0])
        self.mlp_out_layer = nn.Linear(args.q_net_hidden_size, act_size)
        self.GRU_layer = nn.GRUCell(args.q_net_out[0], args.q_net_hidden_size)
        #self.GRU_layer = nn.GRU(args.q_net_out[0], args.q_net_hidden_size, args.q_net_out[1])

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_in_layer.weight)
        nn.init.xavier_uniform_(self.mlp_out_layer.weight)
        #nn.init.xavier_uniform_(self.GRU_layer.all_weights)

    def forward(self, obs_a_cat, hidden_last):
        x = self.mlp_in_layer(obs_a_cat)
        gru_out = self.GRU_layer(x, hidden_last)
        output = self.mlp_out_layer(gru_out)
        return output, gru_out
    
class Hyper_Network(nn.Module):
    def __init__(self, shape_state, args):
        super(Hyper_Network, self).__init__()
        self.w1_layer = nn.Linear(shape_state, args.hyper_par['w1'])
        self.w2_layer = nn.Linear(shape_state, args.hyper_par['w2'])
        self.b1_layer = nn.Linear(shape_state, args.hyper_par['b1'])
        self.b2_layer = nn.Linear(shape_state, args.hyper_par['b2'])
        self.LReLU = nn.LeakyReLU(0.01)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer.weight)

    def forward(self, state):
        w1 = self.w1_layer(state)
        b1 = self.b1_layer(state)
        w2 = self.w2_layer(state)
        b2 = self.b2_layer(state)
        return w1, b1, w2, b2
        
class Mixing_Network(nn.Module):
    def __init__(self, action_size, num_agents, args):
        super(Mixing_Network, self).__init__()
        self.mlp_layer1 = nn.Linear(action_size*num_agents, args.mix_net_out[0])
        self.mlp_layer2 = nn.Linear(args.mix_net_out[0], args.mix_net_out[1])

        self.reset_parameter()
        self.train()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.mlp_layer1.weight)
        nn.init.xavier_uniform_(self.mlp_layer2.weight)
    
    def forward(self, input):
        x = self.mlp_layer1(input)
        output = self.mlp_layer2(x)
        return output