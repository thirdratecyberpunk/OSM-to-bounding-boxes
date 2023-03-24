import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from itertools import count
import random
import os
import cv2
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

from sumo_environment_discrete import SumoEnvironmentDiscrete
from arguments import get_args


class DQN_Linear(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN_Linear, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


if __name__ == '__main__':
    args = get_args()
    # # env = SumoEnvironmentDiscrete('Grid2by2/Grid2by2.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid2by2/Grid2by2_time_variant.rou.xml')

    # env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3_flow0.05.rou.xml')
    env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3_unbalanced.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3_time_variant.rou.xml')


    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    n_actions = env.get_action_space()
    n_obs = env.get_obs_space()
    num_junctions = env.get_num_traffic_lights()
    print('Environment observation space: {}'.format(n_obs))
    print('Environment action space: {}'.format(n_actions))
    print('Number of junctions: {}'.format(num_junctions))

    policy_net = []

    for i in range(num_junctions):
        policy_net.append(DQN_Linear(n_obs, n_actions))
        load_path = 'saved_models/DQN_Grid3by3/DiscreteRL_policy_net_junction' + str(i) + '_flowprob0.05.pt'
        load_policy_model = torch.load(load_path, map_location=lambda storage, loc: storage)
        policy_net[i].load_state_dict(load_policy_model)

        if args.cuda:
            policy_net[i].cuda()

    # print('Demonstration starts. Total episodes: {}'.format(args.num_episodes))
    # for episode in range(args.num_episodes):


    traffic_load_avg = []
    for e in range(5):
        # Play an episode
        obs = env.reset()
        # print('Env reset')
        # rewards_sum = np.zeros([self.num_junctions])
        # rewards_sum = 0
        traffic_load = []
        for t in range(args.episode_length):
            actions_all = []
            for junction_id in range(num_junctions):
                obs_junction_tensor = torch.tensor(obs[junction_id], dtype=torch.float32)
                # obs_junction_tensor = torch.tensor(obs, dtype=torch.float32)
                if args.cuda:
                    obs_junction_tensor = obs_junction_tensor.cuda()

                # action_junction = self._select_action(obs_junction_tensor, junction_id)
                Q_values = policy_net[junction_id](obs_junction_tensor)
                action_tensor = torch.argmax(Q_values)
                action_junction = action_tensor.detach().cpu().numpy().squeeze()
                actions_all.append(action_junction)
            actions_all = np.array(actions_all)

            # env.render()
            obs_new, reward, done, info = env.step(actions_all)
            # rewards_sum += reward
            traffic_load.append(info['traffic_load'])

            obs_last_action = obs[:,0]
            obs_last_action = obs_last_action.flatten()
            # obs_last_action = obs[:self.num_junctions]
            print('[{}]Episode {}, Timestep {}, Obs: {}, Action: {}, Rewards: {}, Traffic load: {}'.format(datetime.now(), e, t, obs_last_action, actions_all, reward, info['traffic_load']))

            obs = obs_new

        # print('[{}] Episode {} finished'.format(datetime.now(), episode))
        traffic_load = np.array(traffic_load)
        print(traffic_load)
        traffic_load_avg.append(traffic_load.mean())
    traffic_load_avg = np.array(traffic_load_avg)
    print(traffic_load_avg)
    print(traffic_load_avg.mean())


    # mpl.style.use('ggplot')
    # fig = plt.figure(1)
    # fig.patch.set_facecolor('white')
    # plt.xlabel('Timestep', fontsize=16)
    # plt.ylabel('Traffic Load', fontsize=16)
    # # plt.title('DQN', fontsize=20)
    # plt.plot(x, traffic_load, color='red', linewidth=2, label='discrete learning, fixed flow')
    # plt.legend(loc='lower left')
    # plt.show()
