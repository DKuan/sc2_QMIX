# File name: model.py
# Author: Zachry
# Time: 2019-12-04
# Description: The enter of this project
import sys

import torch 
import numpy as np
from smac.env import StarCraft2Env

from Q_MIX import QMIX_Agent
from arguments import parse_args

def main(args):
    """ step: init the env and par """
    env = StarCraft2Env(map_name=args.map_name, replay_dir='./replays', replay_prefix='3s5zVS3s6z')
    env_info = env.get_env_info()
    num_agents = env_info["n_agents"]
    shape_obs = env_info['obs_shape'] + 1 # first bit is agent_idx
    shape_state = env_info['state_shape']
    num_actions_set = [env_info["n_actions"]]
    obs_0_idx = np.arange(0, num_agents).reshape(num_agents, 1)

    """ step: init the QMIX agent """
    qmix_agent = QMIX_Agent(shape_obs, shape_state, num_agents, num_actions_set, args)
    qmix_agent.init_trainers(args)

    """ step: interact with the env and learn """
    step_cnt = 0
    done_cnt = 0
    for epi_cnt in range(args.max_episode):
        env.reset()
        episode_reward = 0
        actions_last = env.last_action 
        qmix_agent.memory.create_new_episode()
        hidden_last = np.zeros((num_agents, args.q_net_hidden_size))

        for epi_step_cnt in range(args.per_episode_max_len):
            step_cnt += 1 # update the cnt every time

            # get obs state for select action
            state = env.get_state()
            obs = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
            avail_actions = np.array(env.get_avail_actions())

            # interact with the env and get new state obs
            actions, hidden = qmix_agent.select_actions(avail_actions, obs, actions_last, hidden_last, args)
            reward, done, _ = env.step(actions)
            if epi_step_cnt == args.per_episode_max_len-1: done = True # max len of episode
            state_new = env.get_state()
            obs_new = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
            actions = env.last_action # the env do the things for us

            # update the date and save experience to memory
            if done == True: done_cnt += 1
            qmix_agent.save_memory(obs, actions_last, hidden_last, state, \
                actions, hidden, obs_new, state_new, reward, done)
            actions_last = env.last_action
            hidden_last = hidden

            # agents learn
            qmix_agent.learn(step_cnt, epi_cnt, args)

            # if done, end the episode
            episode_reward += reward
            if done: break

        print("episode_cnt:{} epsilon: {} reward in episode {}".format(epi_cnt, round(qmix_agent.epsilon, 3), episode_reward))

    """ close the env """
    #env.save_replay()
    env.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)

                # hyper_pars_cur = self.hyper_net_cur(state_t_b[step_cnt])
                # hyper_pars_tar = self.hyper_net_tar(state_new_t_b[step_cnt])

                # # cal the qtot_cur and qtot_tar 
                # if b_cnt == 0 and step_cnt == 0:
                #     qtot_tar = self.mixing_net(q_values_tar, hyper_pars_tar)
                #     qtot_cur = self.mixing_net(q_values_cur, hyper_pars_cur)
                # else:
                #     qtot_tar = torch.cat([qtot_tar, self.mixing_net(q_values_tar, hyper_pars_tar)], dim=0) 
                #     qtot_cur = torch.cat([qtot_cur, self.mixing_net(q_values_cur, hyper_pars_cur)], dim=0)

