# File name: enjoy.py
# Author: Zachry
# Time: 2019-12-13
# Description: File for checking the model trained already. 
import os
import sys

import torch 
import numpy as np
from smac.env import StarCraft2Env

from Q_MIX import QMIX_Agent
from arguments import parse_args

def enjoy(env, args):
    """ step1: init the env and par """
    env_info = env.get_env_info()
    num_agents = env_info["n_agents"]
    shape_obs = env_info['obs_shape'] + num_agents # first bit is agent_idx
    shape_state = env_info['state_shape']
    num_actions_set = [env_info["n_actions"]]
    #obs_0_idx = np.arange(0, num_agents).reshape(num_agents, 1)
    obs_0_idx = np.eye(num_agents)
    rewards_list = []

    """ step: init the QMIX agent """
    qmix_agent = QMIX_Agent(shape_obs, shape_state, num_agents, num_actions_set, args)
    qmix_agent.enjoy_trainers(args)

    for _ in range(args.num_epi4evaluation):
        env.reset()
        episode_reward = 0
        actions_last = env.last_action 
        hidden_last = np.zeros((num_agents, args.q_net_hidden_size))
        for epi_step_cnt in range(args.per_episode_max_len):
            # get obs state for select action
            state = env.get_state()
            obs = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
            avail_actions = np.array(env.get_avail_actions())

            # interact with the env and get new state obs
            actions, hidden = qmix_agent.select_actions(avail_actions, obs, actions_last, hidden_last, args, eval_flag=True)
            reward, done, _ = env.step(actions)
            reward = reward*args.reward_scale_par # normalize the reward
            if epi_step_cnt == args.per_episode_max_len-1: done = True # max len of episode

            actions_last = env.last_action
            hidden_last = hidden

            # if done, end the episode
            episode_reward += reward
            if done: break

        # record the reward for final evaluation
        rewards_list.append(episode_reward)

    """ close the env """
    env.save_replay()
    env.close()
    print('The evaluation mean(all/{}) reward is'.format(args.num_epi4evaluation), round(sum(rewards_list)/rewards_list.__len__(), 3))

if __name__ == '__main__':
    args = parse_args()

    env = StarCraft2Env(map_name=args.map_name, replay_dir=os.path.join(os.getcwd(), 'replays'), replay_prefix=args.map_name)

    """ run the main """
    enjoy(env, args)
