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

from sumo_environment_discrete import SumoEnvironmentDiscrete
from arguments import get_args

from visualization import Visualization 
from utils import import_train_configuration, set_sumo, set_train_path, set_top_path

class Greedy_queue_model_agent_discrete:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        self.n_obs = env.get_obs_space()
        self.num_junctions = env.get_num_traffic_lights()
        self.episode_return_all = []
        self.episode_avg_traffic_load_all = []
        print('Environment observation space: {}'.format(self.n_obs))
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.num_junctions))

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'RussianJunctionModels')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def learn(self):
        # env.reset()
        self.episode_return_all = []
        self.episode_avg_traffic_load_all = []
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            # hardcoded assumption that the pretimed agent will rotate every 10 timesteps
            # TODO: parameterise this
            self._duration = 10
            # how many steps this simulation has done
            self._elapsed_duration = 0
            # default to choosing the first action
            self._chosen_action = 0
            if self.args.fullobs:
                obs = self.obs_transfer_full(obs)

            # print('Env reset')
            rewards_sum = np.zeros([self.num_junctions])
            # rewards_sum = 0
            traffic_load_sum = 0
            for t in range(self.args.episode_length):
                self._elapsed_duration = t
                actions_all = []
                for junction_id in range(self.num_junctions):
                    obs_junction_tensor = self._preproc_inputs(obs[junction_id])
                    action_junction = self._select_action(obs_junction_tensor, junction_id)
                    actions_all.append(action_junction)
                actions_all = np.array(actions_all)

                # env.render()
                obs_new, reward, done, info = self.env.step(actions_all)
                rewards_sum += reward
                traffic_load_sum += info['traffic_load']

                reward_overall = reward.sum()
                if self.args.fullobs:
                    obs_new = self.obs_transfer_full(obs_new)

                if self.args.fullobs:
                    index = np.arange(0, self.num_junctions, 1, dtype=int)
                    last_action = obs[index, index * self.n_obs]
                else:
                    last_action = obs[:,0]
                    last_action = last_action.flatten()
                print('[{}] Episode {}, Timestep {}, Last action: {}, Action: {}, Rewards: {}, Traffic load: {}'.format(datetime.now(), episode, t, last_action, actions_all, reward, info['traffic_load']))

                obs = obs_new

            print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, rewards_sum))
            self.episode_return_all.append(rewards_sum)
            self.episode_avg_traffic_load_all.append(traffic_load_sum/self.args.episode_length)
        print('Learning process finished')

    def obs_transfer_full(self, obs):
        obs_full = np.zeros([self.num_junctions, self.num_junctions*self.n_obs], dtype=int)
        for junction_id in range(self.num_junctions):
            obs_full[junction_id, junction_id*self.n_obs:(junction_id+1)*self.n_obs] = obs[junction_id].copy()
        return obs_full

    def _preproc_inputs(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor

    def _select_action(self, state, junction_id):
        """
        Predict the action values from a single state
        In this case, simply rotates between actions after a certain
        amount of time
        """
        return env._get_highest_queue_size_action(junction_id)

if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentDiscrete(rou_file='RussianJunction/osm.rou.xml', 
                                  net_file='RussianJunction/osm.net.xml',
                                    cfg='RussianJunction/osm.sumocfg')
    
    path = set_top_path("yingyi_simulation_results", ['Greedy queue size agent'])

    visualization = Visualization(
        path, 
        dpi=96
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = Greedy_queue_model_agent_discrete(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()

    visualization.plot_single_agent(data=trainer.episode_return_all,
                    filename='reward',
                    xlabel='Episode',
                    ylabel='Episode return value',
                    agent="Pretimed")
    
    visualization.plot_single_agent(data=trainer.episode_avg_traffic_load_all,
                    filename='avg_traffic_load',
                    xlabel='Episode',
                    ylabel='Average traffic load for junction',
                    agent="Pretimed")
