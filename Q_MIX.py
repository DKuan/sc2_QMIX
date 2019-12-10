# File name: Q_MIX.py
# Author: Zachry
# Time: 2019-12-04
# Description: The class of QMIX
import os
import sys

import time
import torch
import random
import numpy as np
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from model import Q_Network, Mixing_Network, Hyper_Network

class QMIX_Agent():
    def __init__(self, shape_obs, shape_state, num_agents, num_actions_set, args):
        self.epsilon = args.epsilon 
        self.shape_obs = shape_obs
        self.shape_state = shape_state
        self.num_agents = num_agents
        self.num_actions_set = num_actions_set
        self.mse_loss = F.mse_loss
        self.memory = ReplayBuffer(args.memory_size)

    def init_trainers(self, args):
        self.q_net_tar = Q_Network(self.shape_obs, max(self.num_actions_set), args).to(args.device)
        self.q_net_cur = Q_Network(self.shape_obs, max(self.num_actions_set), args).to(args.device)
        self.hyper_net = Hyper_Network(self.shape_state, args).to(args.device)
        self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
    
    def save_memory(self, obs, u_last, hidden_last, state, \
                u, hidden, obs_new, state_new, r, done):
        self.memory.add(obs, u_last, hidden_last, state, \
                u, hidden, obs_new, state_new, r, done)

    def select_actions(self, avail_actions, obs, actions_last, hidden_last, args):
        """
        Note:epsilon-greedy to choose the action
        """
        action_all = []
        """ step1: get the q_values """
        q_values, hidden = self.q_net_cur(torch.from_numpy( \
            np.hstack([obs, actions_last])).to(args.device, dtype=torch.float), \
            torch.from_numpy(hidden_last).to(args.device, dtype=torch.float))
        
        """ step2: mask the q_values"""
        mask = torch.from_numpy(avail_actions) # mask the actions
        q_values[mask==0] = float('-inf')
        
        """ choose action by e-greedy """
        avail_act_idxs = [list(np.where(avail_actions[idx]==1)[0]) for idx in range(self.num_agents)]
        avail_actions_random = torch.tensor([random.sample(avail_act_idxs[i], 1) \
            for i in range(self.num_agents)], device=args.device) # all random actions
        avail_actions_random = avail_actions_random.reshape(-1)
        max_actions = torch.max(q_values, dim=1)[1] # all max actions 
        epsilons_choice = torch.rand(max_actions.shape) < self.epsilon # e-greedy choose the idx 
        max_actions[epsilons_choice] = avail_actions_random[epsilons_choice] # exchange the data

        return max_actions.numpy(), hidden.detach().numpy()

    def learn(self, step_cnt, args):
        if step_cnt < args.learning_start_step: return
        if self.epsilon > 0.08 : self.epsilon *= args.anneal_par

        """ step1: get the batch data from the memory """
        obs_t, u_last_t, hidden_last_t, state_t, u_t, hidden_t, \
            obs_new_t, state_new_t, r_t, done_t =  self.memory.sample(args.batch_size)

        """ step2: get the q_values of all agent """
        q_values_tar = self.q_net_tar(obs_new_n, hidden_last)
        q_values_cur = self.q_net_cur(obs_old_n, hidden_last)

        """ step3: init the weight for mixing_network """
        hyper_pars = self.hyper_net(state)

        """ step4: pass the q_values to get Q_tot """
        q_tot_tar = self.mixing_net(q_values_tar)
        q_tot_cur = self.mixing_net(q_values_cur)

        """ step5: cal the loss by bellman equation """
        q_ = r_b*args.gamma + q_tot_tar
        q = q_tot_cur
        loss = self.mse_loss(q_, q)

        """ step5: loss backward """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """ step6: save the model """
        if self.learned_cnt > args.save_model_start and self.learned_cnt % args.save_fre == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            #if not os.path.exists(args.save_path):
            # os.mkdir(args.save_path)
            model_path_now = sys.path.join(args.save_path, time_now)
            os.mkdir(model_path_now) 
            torch.save(self.q_net_tar, os.path.join(model_path_now, time_now))
        return loss