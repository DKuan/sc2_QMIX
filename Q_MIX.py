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
import torch.nn as nn

from replay_buffer import ReplayBuffer
from model import Q_Network, Mixing_Network, Hyper_Network

class QMIX_Agent():
    def __init__(self, shape_obs, shape_state, num_agents, num_actions_set, args):
        self.epsilon = args.epsilon 
        self.shape_obs = shape_obs
        self.shape_state = shape_state
        self.num_agents = num_agents
        self.num_actions_set = num_actions_set
        self.last_cnt4update = 0
        self.last_epi_cnt = 0
        self.mse_loss = F.mse_loss
        self.memory = ReplayBuffer(args.memory_size)
        self.learned_cnt = 0

    def init_trainers(self, args):
        shape_hyper_net = {}
        self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
        self.q_net_tar = Q_Network(self.shape_obs, max(self.num_actions_set), args).to(args.device)
        self.q_net_cur = Q_Network(self.shape_obs, max(self.num_actions_set), args).to(args.device)
        self.hyper_net_tar = Hyper_Network(self.shape_state, self.mixing_net.pars, args).to(args.device)
        self.hyper_net_cur = Hyper_Network(self.shape_state, self.mixing_net.pars, args).to(args.device)
        self.hyper_net_tar.load_state_dict(self.hyper_net_cur.state_dict()) # update the tar net par
        self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par
        # self.optimizer = torch.optim.RMSprop([{'params':self.q_net_cur.parameters()}, 
        #                                         {'params':self.hyper_net_cur.parameters()},
        #     ], lr=args.lr)
        self.params_cur = list(self.q_net_cur.parameters()) + list(self.hyper_net_cur.parameters())
        self.optimizer = torch.optim.RMSprop(self.params_cur, lr=args.lr)
    
    def enjoy_trainers(self, args):
        self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
        self.q_net_cur = torch.load(args.old_model_name+'q_net.pkl', map_location=args.device)
        self.hyper_net_cur = torch.load(args.old_model_name+'hyper_net.pkl', map_location=args.device)

    def save_memory(self, obs_and_u_last, state, \
                u, new_avail_actions, obs_new, state_new, r, done):
        r = np.array([r])[np.newaxis, :]
        done = np.array([done])[np.newaxis, :]
        self.memory.add((obs_and_u_last[np.newaxis, :], state[np.newaxis, :], u[np.newaxis, :], \
            new_avail_actions[np.newaxis, :], obs_new[np.newaxis, :], state_new[np.newaxis, :], r, done))

    def select_actions(self, avail_actions, obs, actions_last, hidden_last, args, eval_flag=False):
        """
        Note:epsilon-greedy to choose the action
        """
        action_all = []
        """ step1: get the q_values """
        q_values, hidden = self.q_net_cur(torch.from_numpy( \
            np.hstack([obs, actions_last])).to(args.device, dtype=torch.float), \
            torch.from_numpy(hidden_last).to(args.device, dtype=torch.float))
        
        """ step2: mask the q_values"""
        mask = torch.from_numpy(avail_actions).to(args.device) # mask the actions
        q_values[mask==0] = float('-inf')
        
        """ choose action by e-greedy """
        avail_act_idxs = [list(np.where(avail_actions[idx]==1)[0]) for idx in range(self.num_agents)]
        avail_actions_random = torch.tensor([random.sample(avail_act_idxs[i], 1) \
            for i in range(self.num_agents)], device=args.device) # all random actions
        avail_actions_random = avail_actions_random.reshape(-1)
        max_actions = torch.max(q_values, dim=1)[1] # all max actions 
        epsilons_choice = torch.rand(max_actions.shape) < self.epsilon # e-greedy choose the idx 
        max_actions[epsilons_choice] = avail_actions_random[epsilons_choice] if eval_flag == False else \
            max_actions[epsilons_choice]# exchange the data

        return max_actions.detach().cpu().numpy(), hidden.detach().cpu().numpy()

    def cal_totq_values(self, batch_data, args):
        """step1: split the batch data and change the numpy data to tensor data """
        obs_and_u_last_n, state_n, u_n, new_avail_act_n, \
            obs_new_n, state_new_n, r_n, done_n =  batch_data # obs_n obs_numpy
        #data_test = np.where(done_n==True)
        obs_and_u_last_t_b = torch.from_numpy(obs_and_u_last_n).to(args.device, dtype=torch.float) # obs_tensor_batch 
        state_t_b = torch.from_numpy(state_n).to(args.device, dtype=torch.float) 
        u_t_b = torch.from_numpy(u_n).to(args.device, dtype=torch.float)
        obs_new_t_b = torch.from_numpy(obs_new_n).to(args.device, dtype=torch.float)
        new_avail_act_t_b = torch.from_numpy(new_avail_act_n).to(args.device, dtype=torch.uint8)
        state_new_t_b = torch.from_numpy(state_new_n).to(args.device, dtype=torch.float) 
        r_t_b = torch.from_numpy(r_n.squeeze(-1)).to(args.device, dtype=torch.float) 
        done_t_b = torch.from_numpy(~done_n.squeeze(-1)).to(args.device, dtype=torch.float) 

        """step2: cal the totq_values """
        q_cur = None # record the totq_cur values
        q_tar = None
        step_cnt = 0

        # cal the q_cur and q_tar
        for b_cnt in range(args.batch_size):
            #print('the step cnt is ', step_cnt)
            hidden_cur = torch.zeros((self.num_agents, args.q_net_hidden_size), device=args.device)
            hidden_tar = torch.zeros((self.num_agents, args.q_net_hidden_size), device=args.device)
            while True:
                q_values_cur, hidden_cur = self.q_net_cur(obs_and_u_last_t_b[step_cnt], hidden_cur)
                q_values_tar, hidden_tar = self.q_net_tar( \
                    torch.cat((obs_new_t_b[step_cnt], u_t_b[step_cnt]), dim=-1), hidden_tar)
                q_cur = q_values_cur[u_t_b[step_cnt].to(torch.uint8)].reshape(1, 1, -1) if b_cnt==0 and step_cnt == 0 else \
                    torch.cat([q_cur, q_values_cur[u_t_b[step_cnt].to(torch.uint8)].reshape(1, 1, -1)], dim=0)
                q_values_tar[~new_avail_act_t_b[step_cnt]] = float('-inf') # get the Q-values for max_Q 
                q_tar = torch.max(q_values_tar, dim=-1)[0].reshape(1, 1, -1) if b_cnt==0 and step_cnt == 0 else \
                    torch.cat([q_tar, torch.max(q_values_tar, dim=-1)[0].reshape(1, 1, -1)], dim=0)

                # update the flag and data
                done = done_t_b[step_cnt]
                step_cnt += 1 # udpate the cnt
                if not done: break
        
        # cal the qtot_cur and qtot_tar 
        hyper_pars_cur = self.hyper_net_cur(state_t_b)
        hyper_pars_tar = self.hyper_net_tar(state_new_t_b)
        qtot_tar = self.mixing_net(q_tar, hyper_pars_tar) # the net is no par
        qtot_tar = r_t_b + qtot_tar * args.gamma * done_t_b
        qtot_cur = self.mixing_net(q_cur, hyper_pars_cur)

        return qtot_cur, qtot_tar
        
    def learn(self, step_cnt, epi_cnt, args):
        loss = 0.0 
        if epi_cnt < args.learning_start_episode: return
        #if step_cnt % args.learning_fre != 0: return
        if epi_cnt <= self.last_epi_cnt: return
        self.last_epi_cnt = epi_cnt
        if self.epsilon > 0.08 : self.epsilon -= args.anneal_par
        if epi_cnt % args.learning_fre != 0: return
        #print('learn cnt ', self.learned_cnt)
        self.learned_cnt += 1

        """ step1: get the batch data from the memory and change to tensor"""
        batch_data =  self.memory.sample(args.batch_size) # obs_n obs_numpy
        q, q_ = self.cal_totq_values(batch_data, args)

        """ step5: cal the loss by bellman equation """
        loss = self.mse_loss(q_.detach(), q)

        """ step5: loss backward """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net_cur.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.hyper_net_cur.parameters(), args.max_grad_norm)
        self.optimizer.step()

        """ step : update the tar and cur network """
        if epi_cnt > args.learning_start_episode and \
            epi_cnt > self.last_cnt4update and \
            (epi_cnt - self.last_cnt4update)%args.tar_net_update_fre == 0:
            self.last_cnt4update = epi_cnt
            self.hyper_net_tar.load_state_dict(self.hyper_net_cur.state_dict()) # update the tar net par
            self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par

        """ step6: save the model """
        if self.learned_cnt > args.start_save_model and self.learned_cnt % args.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            model_path_now = os.path.join(args.save_dir, time_now)
            os.mkdir(model_path_now) 
            torch.save(self.q_net_tar, os.path.join(model_path_now, 'q_net.pkl'))
            torch.save(self.hyper_net_tar, os.path.join(model_path_now, 'hyper_net.pkl'))
            print('save the model in time:{}'.format(time_now))
        return loss

    
