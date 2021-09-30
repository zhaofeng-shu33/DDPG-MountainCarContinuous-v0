# -*- coding: utf-8 -*-
"""
@author: zhaofeng-shu33

A gym env to simulate ODE Solver
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from scipy.integrate._ivp.rk import rk_step
from scipy.integrate._ivp.rk import RK23

class SpiralODE_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        self.tmax = 2 * np.pi
        self.theta = 1.0
        self.U_MIN = 0.001
        self.U_MAX = 10.0
        self.state = np.array([self.U_MIN, self.U_MIN]) # u_0
        self.t = 0.0
        self.max_action = self.tmax # maximal h
        self.solver = RK23(self.ode_func, self.t, self.state, )

        self.seed()
        self.reset()

    def ode_func(self, t, y):
        # theta: parameter of the ODE function
        return [self.theta[0] * np.cos(t) - y[1] + self.u0[1], self.theta[0] * np.sin(t) + y[0] - self.u0[0]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # currently use ODE23.
        h = action[0]
        current_f = self.ode_func(self.t, self.state)
        y_new, f_new = rk_step(self.ode_func, self.t, self.state, current_f, h, )
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity])
        self.state[0] = ((self.state[0] + 1.2) / 1.8) * 2 - 1
        self.state[1] = ((self.state[1] + 0.07) / 0.14) * 2 - 1
        return self.state, reward, done, {}

    def reset(self):
        self.u0 = self.np_random.uniform(low=self.U_MIN, high=self.U_MAX, size=2)
        self.state = np.copy(self.u0)
        self.t = 0.0
        # return np.array(self.state)

