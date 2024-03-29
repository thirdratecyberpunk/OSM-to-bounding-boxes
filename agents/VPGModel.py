# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:19:05 2021

@author: Lewis
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import sys

class PolicyNetwork(nn.Module):
    """
    Neural network that attempts to learn underlying policy mapping
    states to actions
    """
    def __init__(self, in_features=48, num_actions=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))
        return x

def vector_to_tensor(vector):
    return torch.from_numpy(vector).float()

# defining an agent who utilises Vanilla Policy Gradient
# agent attempts to directly learn the policy function mapping S to A
# need to therefore directly optimise the policy function
# can learn determinisitic and stochastic policies
# capable of handling Partially Observable Markov Decision Processes
class VPGAgent():
    # initialises
    def __init__(self, input_dim=80, output_dim=4, epsilon=0.05, alpha=0.1, gamma=0.5):
        # checking CUDA support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # gamma value
        self.gamma = gamma
        # policy network
        self.policy_network = PolicyNetwork(input_dim, output_dim).to(self.device)
        self.rewards = []
        self.actions = []
        self.states = []
        # optimiser
        self.optimiser = torch.optim.Adam(self.policy_network.parameters())
        # loss
        self.pseudo_loss = 0
        
    def choose_action(self, current_state):
        """Returns the optimal action for the state from the policy"""
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        # gets the vector of probabilities from the neural network
        probabilities = self.policy_network(current_state_tensor)
        # samples from available action space using probability distribution
        # if the agent has finished an episode, fix gradients
        sampler = Categorical(probabilities)
        chosen_action = sampler.sample()     
        return chosen_action

    def get_values_from_policy_network(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        return self.policy_network(current_state_tensor)

    def add_values_to_collection(self, old_state, reward, new_state, action):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(torch.from_numpy(new_state).float().to(self.device))

    def finish_episode(self):
        # calculating the gradient at the end of an episode
        # preprocess rewards
        self.rewards = np.array(self.rewards)
        R = torch.tensor([np.sum(self.rewards[i:]*(self.gamma**np.array(range(i, len(self.rewards))))) for i in range(len(self.rewards))]).to(self.device)
        # preprocess states and actions
        states_tensor = torch.stack(self.states).to(self.device)
        #torch.tensor(self.states).float().to(self.device)
        actions_tensor = torch.tensor(self.actions).to(self.device)
        
        probs = self.policy_network(states_tensor).to(self.device)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(actions_tensor)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
        self.pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
        # update policy weights
        self.optimiser.zero_grad()
        self.pseudo_loss.backward()
        self.optimiser.step()
        # clear 
        self.rewards = []
        self.actions = []
        self.states = []

    def save_agent(self, agent_name, epoch):
        torch.save({
            'epoch': epoch,
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'pseudo_loss': self.pseudo_loss,
            'epoch' : epoch
            }, agent_name)

    def load_agent(self, path):
        print("Before loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.pseudo_loss = checkpoint["pseudo_loss"]
        
        print("After loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        return checkpoint["epoch"]

class TrainModel:
    def __init__(self, input_dim, output_dim, gamma, epsilon, alpha):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._gamma = gamma
        self._epsilon = epsilon
        self._alpha = alpha
        self._model = self._build_model(output_dim)


    def _build_model(self, output_dim):
        """
        Build and compile a fully connected deep neural network
        """
        # input_dim=80, output_dim=4, epsilon=0.05, alpha=0.1, gamma=0.5
        agent = VPGAgent(self._input_dim, output_dim, self._epsilon, self._alpha, self._gamma)
        return agent    

    def choose_action(self, current_state):
        return self._model.choose_action(current_state)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.get_values_from_policy_network(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.get_values_from_policy_network(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        # def learn(self, old_state, reward, new_state, action):
        self._model.learn_batch(states, q_sa)


    def save_model(self, path, episode):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save_agent(f"{path}/VPGLearningAgent_{episode}", episode)
     
    def after_action(self, old_state, old_action, reward, current_state):
        """
        

        Parameters
        ----------
        old_state : TYPE
            DESCRIPTION.
        old_action : TYPE
            DESCRIPTION.
        reward : TYPE
            DESCRIPTION.
        current_state : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._model.add_values_to_collection(old_state, reward, current_state, old_action)
     
    def finish_episode(self, training_epochs):
        self._model.finish_episode()
     
    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def agent_name(self):
        return type(self._model).__name__

class TestModel:
    def __init__(self, input_dim, output_dim,  model_path):
        self._input_dim = input_dim
        self._model = VPGAgent(self._input_dim, output_dim)
        self._model.load_agent(model_path)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.get_values_from_policy_network(state)


    @property
    def input_dim(self):
        return self._input_dim