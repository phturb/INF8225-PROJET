import gym
import numpy as np
import random
from PIL import Image
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

from projet.agent import Agent

from copy import deepcopy
from collections import deque


class DQNv3(Agent):
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000,
                 epsilon_decay=0.999, epsilon_min=0.10, gamma=0.89, trials=5000, batch_size=64, lr=0.001):
        super().__init__(env, model_factory, input_shape, batch_states_process,
                         observation_process, reward_process, memory_size)
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.10
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model = None
        self.target_model = None
        self.init_models()

    def save_models(self, model_name_prefix):
        self.model.save(model_name_prefix)

    def load_model(self, path="success.model"):
        self.model = load_model(path)
        self.target_model = load_model(path)
        self.model.summary()

    def action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if random.random() <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state)[0])
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, reward):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            if done:
                dones.append(0)
            else:
                dones.append(1)
        states = self.batch_states_process(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = self.batch_states_process(np.array(next_states))
        dones = np.array(dones)
        Q_targets = self.target_model.predict_on_batch(next_states)
        Q_targets = np.max(Q_targets, axis=1).flatten()
        Q_targets *= self.gamma
        Q_targets *= dones
        Rs = rewards + Q_targets
        targets = self.target_model.predict_on_batch(states)
        for _, (target, R, action) in enumerate(zip(targets, Rs, actions)):
            target[action] = R
        self.model.train_on_batch(states, targets)
        self.target_update()

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def init_models(self):
        self.model = self.create_model(
            self.input_shape, self.env.action_space.n)
        self.model.summary()
        self.target_model = self.create_model(
            self.input_shape, self.env.action_space.n)
        self.model.compile(loss="mean_squared_error", optimizer=Adam(
            lr=self.learning_rate, epsilon=1.5*10e-4, clipnorm=1.0))
        self.target_model.compile(loss="mean_squared_error", optimizer=Adam(
            lr=self.learning_rate, epsilon=1.5*10e-4, clipnorm=1.0))
        self.target_update()
