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

class DQN_Linear_discrete(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN_Linear_discrete, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


class DQN_Linear(nn.Module):
    def __init__(self, n_obs, n_actions, n_junctions):
        super(DQN_Linear, self).__init__()
        self.n_obs_discrete = int(n_obs/n_junctions)

        self.fc1 = nn.Linear(self.n_obs_discrete, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        self.fc4 = nn.Linear(self.n_obs_discrete, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.fc7 = nn.Linear(self.n_obs_discrete, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)

        self.fc10 = nn.Linear(self.n_obs_discrete, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.head1 = nn.Linear(256*n_junctions, n_actions**n_junctions)


    def forward(self, input):
        all = []

        x1 = F.relu(self.fc1(input[..., : self.n_obs_discrete]))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        all.append(x1)

        x2 = F.relu(self.fc4(input[..., self.n_obs_discrete : self.n_obs_discrete*2]))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))
        all.append(x2)

        x3 = F.relu(self.fc7(input[..., self.n_obs_discrete*2 : self.n_obs_discrete*3]))
        x3 = F.relu(self.fc8(x3))
        x3 = F.relu(self.fc9(x3))
        all.append(x3)

        x4 = F.relu(self.fc10(input[..., self.n_obs_discrete*3:]))
        x4 = F.relu(self.fc11(x4))
        x4 = F.relu(self.fc12(x4))
        all.append(x4)

        all = torch.cat(all,-1)
        return self.head1(all)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_agent_joint_state_preload:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        self.n_obs = env.get_obs_space()
        self.num_junctions = env.get_num_traffic_lights()
        print('Environment observation space: {}'.format(self.n_obs))
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.num_junctions))

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'DQN_Grid2by2')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.policy_net = DQN_Linear(self.n_obs, self.n_actions, self.num_junctions)
        self.target_net = DQN_Linear(self.n_obs, self.n_actions, self.num_junctions)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = self.args.lr)
        self.buffer = ReplayBuffer(self.args.buffer_size)

        # for i in range(self.num_junctions):
        policy_net_model = self.policy_net.state_dict()
        # n_obs_discrete = int(self.n_obs/self.num_junctions)
        # discrete_net = DQN_Linear_discrete(n_obs_discrete, self.n_actions)
        discrete_0_model = torch.load(self.save_path + '/DiscreteRL_policy_net_junction0_flowprob0.005.pt', map_location=lambda storage, loc: storage)
        discrete_1_model = torch.load(self.save_path + '/DiscreteRL_policy_net_junction1_flowprob0.005.pt', map_location=lambda storage, loc: storage)
        discrete_2_model = torch.load(self.save_path + '/DiscreteRL_policy_net_junction2_flowprob0.005.pt', map_location=lambda storage, loc: storage)
        discrete_3_model = torch.load(self.save_path + '/DiscreteRL_policy_net_junction3_flowprob0.005.pt', map_location=lambda storage, loc: storage)

        policy_net_model['fc1.weight'].copy_(discrete_0_model['fc1.weight'].data)
        policy_net_model['fc1.bias'].copy_(discrete_0_model['fc1.bias'].data)
        policy_net_model['fc2.weight'].copy_(discrete_0_model['fc2.weight'].data)
        policy_net_model['fc2.bias'].copy_(discrete_0_model['fc2.bias'].data)
        policy_net_model['fc3.weight'].copy_(discrete_0_model['fc3.weight'].data)
        policy_net_model['fc3.bias'].copy_(discrete_0_model['fc3.bias'].data)

        policy_net_model['fc4.weight'].copy_(discrete_1_model['fc1.weight'].data)
        policy_net_model['fc4.bias'].copy_(discrete_1_model['fc1.bias'].data)
        policy_net_model['fc5.weight'].copy_(discrete_1_model['fc2.weight'].data)
        policy_net_model['fc5.bias'].copy_(discrete_1_model['fc2.bias'].data)
        policy_net_model['fc6.weight'].copy_(discrete_1_model['fc3.weight'].data)
        policy_net_model['fc6.bias'].copy_(discrete_1_model['fc3.bias'].data)

        policy_net_model['fc7.weight'].copy_(discrete_2_model['fc1.weight'].data)
        policy_net_model['fc7.bias'].copy_(discrete_2_model['fc1.bias'].data)
        policy_net_model['fc8.weight'].copy_(discrete_2_model['fc2.weight'].data)
        policy_net_model['fc8.bias'].copy_(discrete_2_model['fc2.bias'].data)
        policy_net_model['fc9.weight'].copy_(discrete_2_model['fc3.weight'].data)
        policy_net_model['fc9.bias'].copy_(discrete_2_model['fc3.bias'].data)

        policy_net_model['fc10.weight'].copy_(discrete_3_model['fc1.weight'].data)
        policy_net_model['fc10.bias'].copy_(discrete_3_model['fc1.bias'].data)
        policy_net_model['fc11.weight'].copy_(discrete_3_model['fc2.weight'].data)
        policy_net_model['fc11.bias'].copy_(discrete_3_model['fc2.bias'].data)
        policy_net_model['fc12.weight'].copy_(discrete_3_model['fc3.weight'].data)
        policy_net_model['fc12.bias'].copy_(discrete_3_model['fc3.bias'].data)
        # for name, param in policy_net_model.items():
        #     print(name)
        #     print(param)
        self.policy_net.load_state_dict(policy_net_model)
        self.policy_net.eval()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.args.cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

    def learn(self):
        # env.reset()
        episode_return_all = []
        episode_avg_traffic_load_all = []
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            # print('Env reset')
            reward_sum = 0
            traffic_load_sum = 0
            for t in range(self.args.episode_length):
                # Play a frame
                # Variabalize 210, 160
                obs_tensor = self._preproc_inputs_joint(obs)
                # obs_tensor_all.append(obs_tensor)
                # if len(self.buffer) < self.args.batch_size:
                #     action = self._select_action_random()
                # else
                action_joint = self._select_action_joint(obs_tensor)
                action_processed = self._action_postpro(action_joint)
                # action = self._select_action_inorder(t)
                # action = self._select_action_random()
                # print('Action selected: {}'.format(action))
                # env.render()
                obs_new, reward, done, info = self.env.step(action_processed)
                reward_sum += reward
                traffic_load_sum += info['traffic_load']
                # calculate the reward for junction 1 anf junction 2

                # Store episode data into the buffers
                self.buffer.push(obs, action_joint, obs_new, reward)

                obs_last_action = []
                for i in range(self.num_junctions):
                    obs_last_action.append(obs[13 * i])
                obs_last_action = np.array(obs_last_action)
                print('[{}] Episode {}, Timestep {}, Obs: {},  Action network: {}, Action: {}, Rewards: {}, Traffic load: {}' \
                        .format(datetime.now(), episode, t, obs_last_action, action_joint, action_processed, reward, info['traffic_load']))

                obs = obs_new

                # Train the networks
                #print('Optimizing starts')
                if len(self.buffer) >= self.args.batch_size:
                    # print('Updating policy network')
                    self._optimize_model(self.args.batch_size)
                #print('Optimizing finishes')

                 # Update the target network, copying all weights and biases in DQN
                if t >= self.args.target_update_step and t % self.args.target_update_step == 0:
                    print('Updating target networks')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # if done:
                #     break

            print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, reward_sum))
            episode_return_all.append(reward_sum)
            episode_avg_traffic_load_all.append(traffic_load_sum/self.args.episode_length)
            np.save(self.save_path + '/CentralizedRL_state_preload_episode_return_queue_reward_flowprob0.005.npy', episode_return_all)
            np.save(self.save_path + '/CentralizedRL_state_preload_avg_traffic_load_queue_reward_flowprob0.005.npy', episode_avg_traffic_load_all)

            torch.save(self.policy_net.state_dict(), self.save_path + '/CentralizedRL_state_preload_policy_net_flowprob0.005.pt')

        print('Learning process finished')

    def _preproc_inputs_joint(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor

    def _select_action_joint(self, state):
        # global steps_done
        sample = random.random()
        # eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
        #     math.exp(-1. * steps_done / self.args.eps_decay)
        # steps_done += 1
        if sample > self.args.random_eps:
            with torch.no_grad():
                Q_values_joint = self.policy_net(state)
                # print(Q_values)
                # take the Q_value index with the largest expected return
                # action_tensor = Q_values.max(1)[1].view(1, 1)
                action_tensor = torch.argmax(Q_values_joint)
                action_joint = action_tensor.detach().cpu().numpy().squeeze().item()
                # print(action_tensor)
                return action_joint
        else:
            return random.randrange(self.n_actions ** self.num_junctions)

    def _action_postpro(self, action_joint):
        actions = []
        remain = action_joint
        for junction_id in range(self.num_junctions):
            actions.append(remain % self.n_actions)
            remain = int(remain / self.n_actions)
        # actions.append(action_joint % self.n_actions)
        # actions.append(int(action_joint / self.n_actions))
        actions = np.array(actions)
        return actions

    # def _select_action_random(self):
    #     return random.randrange(self.n_actions ** self.args.num_junctions)
    #
    # def _select_action_inorder(self, timestep):
    #     action = (timestep % (self.n_actions ** self.args.num_junctions))
    #     return action

    def _optimize_model(self, batch_size):
        states, actions, next_states, rewards = Transition(*zip(*self.buffer.sample(batch_size)))

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        if self.args.cuda:
            states = states.cuda()
            next_states = next_states.cuda()
            rewards = rewards.cuda()
            actions = actions.cuda()


        predicted_values = torch.gather(self.policy_net(states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_values = rewards + self.args.gamma * next_state_values

        # Compute loss
        loss = F.smooth_l1_loss(predicted_values, expected_values)

        # calculate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return

if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentCentralize()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_agent_joint_state_preload(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()
