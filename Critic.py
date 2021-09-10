from torch import nn
import torch

class CriticModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticModel, self).__init__()
        # Add hidden layer(s) for state pathway
        # net = layers.Dense(units=20, activation='relu')(states)
        self.net_1 = nn.Sequential(nn.Linear(state_size, 20, dtype=torch.float64), nn.ReLU())
        # net = layers.Add()([net, actions])
        # net = layers.Dense(units=20, activation='relu')(net)
        self.net_2 = nn.Sequential(nn.Linear(20, 20, dtype=torch.float64), nn.ReLU())

        # lin_states = layers.Dense(units=20, activation='relu')(states)
        self.lin_states = nn.Sequential(nn.Linear(state_size, 20, dtype=torch.float64), nn.ReLU())
        # net = layers.Add()([net, lin_states])

        # Add final output layer to produce action values (Q values)
        # Q_values = layers.Dense(units=1,name='q_values')(net)
        self.Q_values = nn.Linear(20, 1, dtype=torch.float64)

    def forward(self, states, actions):
        net = self.net_1.forward(states)
        net = net + actions # merge layer
        net = self.net_2(net)
        _lin_states = self.lin_states(states)
        net = net + _lin_states
        return self.Q_values(net)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """
        Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values.
        """
        # Define input layers
        # states = layers.Input(shape=(self.state_size,), name='states')
        # actions = layers.Input(shape=(self.action_size,), name='actions')


        # Create Keras model
        self.model = CriticModel(self.state_size, self.action_size)

        # Define optimizer and compile model for training with
        # built-in loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

    def update_actor_parameters(self, states, actions_tensor):
        q_values = self.model.forward(torch.from_numpy(states), actions_tensor)
        loss = -1.0 * torch.mean(q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_on_batch(self, x, y):
        states = torch.from_numpy(x[0])
        actions = torch.from_numpy(x[1])
        q_target = torch.from_numpy(y)
        q_values = self.model.forward(states, actions)
        loss_val = self.loss(q_values, q_target) # y is q_target
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()