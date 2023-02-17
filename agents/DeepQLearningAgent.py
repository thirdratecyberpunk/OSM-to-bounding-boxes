import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    # TODO: make in_features not hardcoded based on environment size
    # could add a method to get state size in environments?
    def __init__(self, in_features=80, num_actions=4):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def vector_to_tensor(vector):
    return torch.from_numpy(vector).float()

# defining an agent who utilises DEEP Q-learning
# rather than utilise a Q-table to store all state-reward pairs
# uses a neural network to learn a distribution
# takes state as input, generates Q-value for all possible actions as output
class DeepQLearningAgent():
    def __init__(self, epsilon=0.05, alpha=0.1, gamma=1, batch_size=128, in_features=80, possible_actions=3):
        # checking CUDA support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # % chance that the agent will perform a random exploratory action
        self.epsilon = epsilon
        # learning rate -> how much the difference between new values is changed
        self.alpha = alpha
        # discount factor -> used to balance future/immediate reward
        self.gamma = gamma
        self.in_features = in_features
        # neural network outputs Q(state, action) for all possible actions
        # for gridworld, all actions are UP, DOWN, LEFT, RIGHT
        # therefore output should contain 4 outputs
        self.possible_actions = possible_actions
        # policy network
        self.policy_qnn = DQN(in_features).to(self.device)
        # neural network that acts as the Q function approximator
        self.target_qnn = DQN(in_features).to(self.device)
        # loss function
        self.loss_function = nn.MSELoss()
        # loss value
        self.loss = 0
        # optimiser
        # self.optimiser = optim.SGD(self.policy_qnn.parameters(),lr=0.001,momentum=0.9)
        self.optimiser = optim.Adam(self.policy_qnn.parameters())
        # size of a batch of replays to sample from
        self.batch_size = batch_size

    def get_q_values_from_policy_network(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        return self.policy_qnn.forward(current_state_tensor)

    def get_q_values_from_target_network(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        return self.target_qnn.forward(current_state_tensor)

    def choose_action(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        """Returns the optimal action for the state from the Q value as predicted by the neural network"""
        if np.random.uniform(0,1) < self.epsilon:
            # chooses a random exploratory action if chance is under epsilon
            action = np.random.choice(range(self.possible_actions))
        else:
            # gets the values associated with action states from the neural network
            q_values = self.policy_qnn.forward(current_state_tensor)
            q_values_for_states = dict(zip(range(self.possible_actions), (x.item() for x in q_values)))
            # chooses the action with the best known 
            action = sorted(q_values_for_states.items(), key=lambda x: x[1])[0][0]
        return action

    def learn_batch(self, states, q_values):
        # calculate expected Q values for given states
        states_tensor = torch.tensor(states).float().to(self.device)
        q_values_tensor = torch.tensor(q_values).float().to(self.device)
        old_state_q_values = self.policy_qnn(states_tensor).to(self.device)
        q_values = torch.tensor(q_values).to(self.device)
        # calculate loss based on difference between expected and actual values
        self.optimiser.zero_grad()
        self.loss = self.loss_function(old_state_q_values, q_values_tensor)
        self.loss.backward()
        self.optimiser.step()

    def finish_episode(self):
        """
        Updates the weights of the target network

        Returns
        -------
        None.

        """
        print("Updating target network weights...")
        self.target_qnn.load_state_dict(self.policy_qnn.state_dict())
    
    def save_agent(self, agent_name, epoch):
        torch.save({
            'epoch': epoch,
            'policy_model_state_dict': self.policy_qnn.state_dict(),
            'target_model_state_dict': self.target_qnn.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'loss': self.loss,
            'epoch' : epoch
            }, agent_name)

    def load_agent(self, path):
        print("Before loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        
        checkpoint = torch.load(path)
        self.policy_qnn.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_qnn.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.loss = checkpoint["loss"]
        
        print("After loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        return checkpoint["epoch"]