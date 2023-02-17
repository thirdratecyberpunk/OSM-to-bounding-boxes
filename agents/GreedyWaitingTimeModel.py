# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 00:14:37 2021

@author: Lewis
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:30:56 2021

@author: Lewis
"""
import numpy as np

class GreedyWaitingTimeModel:
    def __init__(self, tlc):
        self._tlc = tlc

    def _build_model(self, num_layers, output_dim):
        pass
    
    def choose_action(self, current_state):
        return self.predict_one(current_state)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        In this case, chooses the action corresponding to the highest pressure
        lane
        """
        action = self._tlc._get_greedy_total_wait_time_action()
        return action


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        pass


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        pass


    def save_model(self, path, episode):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        pass
       
    def after_action(self, old_state, old_action, reward, current_state):
        pass
     
        
    def finish_episode(self, training_epochs):
        pass

    @property
    def agent_name(self):
        return "GreedyWaitingTimeModel"

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, tlc):
        self._tlc = tlc

    def predict_one(self, state):
        """
        Predict the action values from a single state
        In this case, chooses the action corresponding to the highest pressure
        lane
        """
        action = self._tlc._get_greedy_total_wait_time_action()
        return action
    
    @property
    def input_dim(self):
        return self._input_dim