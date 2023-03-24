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

from sumo_environment_centralise import SumoEnvironmentCentralize
from arguments import get_args


class DQN_Linear(nn.Module):
    def __init__(self, n_obs, n_actions, n_junctions):
        super(DQN_Linear, self).__init__()
        self.fc1 = nn.Linear(n_obs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.head = nn.Linear(512, n_actions ** n_junctions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

def action_postpro(action_joint, num_junctions, n_actions):
    actions = []
    remain = action_joint
    for junction_id in range(num_junctions):
        actions.append(remain % n_actions)
        remain = int(remain / n_actions)
    actions = np.array(actions)
    return actions


if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentCentralize()

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

    policy_net = DQN_Linear(n_obs, n_actions, num_junctions)
    load_path = 'saved_models/DQN_Grid2by2/Centralized_LfD_policy_net.pt'
    load_policy_model = torch.load(load_path, map_location=lambda storage, loc: storage)
    policy_net.load_state_dict(load_policy_model)

    if args.cuda:
        policy_net.cuda()

    print('Demonstration starts. Total episodes: {}'.format(args.num_episodes))
    for episode in range(args.num_episodes):
        # Play an episode
        obs = env.reset()
        # print('Env reset')
        reward_sum = 0
        traffic_load_sum = 0
        for t in range(args.episode_length):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            if args.cuda:
                obs_tensor = obs_tensor.cuda()

            with torch.no_grad():
                Q_values_joint = policy_net(obs_tensor)
                action_tensor = torch.argmax(Q_values_joint)
                action_joint = action_tensor.detach().cpu().numpy().squeeze().item()

            action_processed = action_postpro(action_joint, num_junctions, n_actions)
            obs_new, reward, done, info = env.step(action_processed)
            reward_sum += reward
            traffic_load_sum += info['traffic_load']

            print('[{}] Episode {}, Timestep {}, Obs: {},  Action network: {}, Action: {}, Rewards: {}, Traffic load: {}' \
                    .format(datetime.now(), episode, t, obs[:num_junctions], action_joint, action_processed, reward, info['traffic_load']))

            obs = obs_new

        print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, reward_sum))

    print('Demonstration finished')
