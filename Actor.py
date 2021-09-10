# from keras import layers, models, optimizers, initializers, regularizers
# from keras import backend as K
# from tensorflow.python.util.tf_inspect import Parameter
from torch import nn
import torch

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """
        Build an actor (policy) network that maps states -> actions.
        * In the MountainCar scenario there is no need to adjust the
          action range because the tanh function output maps [-1,1]
        """
        # Define input layer (states)
        # states = layers.Input(shape=(self.state_size,), name='states')

        
        # net = layers.Dense(units=40, activation='relu')(states)
        # net2 = layers.Dense(units=20, activation='relu')(states)

        # net = layers.Add()([net1, net2])

        # net = layers.Dense(units=20, activation='relu')(net)

        # final output layer
        # actions = layers.Dense(units=self.action_size, activation='tanh',
        #                        name='raw_actions')(net)
        self.model = nn.Sequential(nn.Linear(self.state_size, 40, dtype=torch.float64), nn.ReLU(),
                                   nn.Linear(40, 20, dtype=torch.float64), nn.ReLU(),
                                   nn.Linear(20, self.action_size, dtype=torch.float64), nn.Tanh())
        # Create Keras model
        # self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # action_gradients = layers.Input(shape=(self.action_size,))
        # loss = K.mean(-action_gradients * actions)
        
        # Define optimizer and training function
        # optimizer = optimizers.Adam(lr=0.0001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # updates_op = optimizer.get_updates(m1
        #     params=self.model.trainable_weights, loss=loss)

    def train_fn(self, states, action_gradients):
        # action_gradients should not contain any gradient infomation
        actions = self.model(torch.from_numpy(states))
        loss = torch.mean(-torch.from_numpy(action_gradients) * actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.train_fn = K.function(
        #    inputs=[self.model.input, action_gradients, K.learning_phase()],
        #    outputs=[],
        #    updates=updates_op)
