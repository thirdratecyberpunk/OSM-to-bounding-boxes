# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:19:05 2021

@author: Lewis
"""
import numpy as np

from agents.VPGAgent import VPGAgent

# TODO: remove TrainModel class, this abstraction isn't helpful anymore

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