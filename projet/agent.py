import gym
from collections import deque
from copy import deepcopy
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten
from keras.models import Sequential, load_model
from keras import Model
from PIL import Image
import random
import numpy as np


class Agent():
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000):
        self.env = env
        self.memory = deque(maxlen=memory_size)
        self.model_trained = False
        assert model_factory is not None
        self.create_model = model_factory
        self.batch_states_process = batch_states_process
        self.observation_process = observation_process
        self.reward_process = reward_process
        if input_shape is None:
            self.input_shape = self.env.observation_space.shape
        else:
            self.input_shape = input_shape

    def load_model(self, path="success.model"):
        raise NotImplementedError()

    def action(self, state):
        raise NotImplementedError()

    def replay(self, reward):
        raise NotImplementedError()

    def save_models(self, model_name_prefix):
        raise NotImplementedError()

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def execute_action(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.observation_process(deepcopy(next_state))
        if self.reward_process is not None:
            reward = self.reward_process(reward)
        return next_state, reward, done, info

    def train(self, trials=5000, trial_size=250, model_name="success.model", min_trials=100, render=False,
              solve_condition=lambda total_rewards_list, trial, min_trials: trial >= 100 and sum(total_rewards_list)/len(total_rewards_list) >= -110):
        self.total_rewards_list = deque(maxlen=min_trials)
        for trial in range(trials):
            current_state = self.observation_process(
                deepcopy(self.env.reset()))
            total_rewards = 0
            for step in range(trial_size):
                if render:
                    self.env.render()
                action = self.action(current_state)
                next_state, reward, done, _ = self.execute_action(action)
                self.replay(reward)
                current_state = next_state
                total_rewards += reward
                if done:
                    self.execute_action(self.action(current_state))
                    self.replay(0.)
                    break
            self.total_rewards_list.append(total_rewards)
            if step >= self.env.spec.max_episode_steps - 1:
                print("Failed to complete in trial {}".format(
                    trial), total_rewards)
            else:
                print("Completed in {} trials".format(trial), total_rewards)
            if trial % 100 == 0:
                self.save_models(model_name)
            if solve_condition(self.total_rewards_list, trial, min_trials):
                print("Solved in {} trials".format(trial), total_rewards)
                break
        self.save_models(model_name)

    def test(self, trials=5, trial_size=250, render=False):
        for t in range(trials):
            current_state = self.observation_process(
                deepcopy(self.env.reset()))
            total_rewards = 0
            for _ in range(trial_size):
                if render:
                    self.env.render()
                action = self.action(current_state)
                current_state, reward, done, _ = self.execute_action(action)
                total_rewards += reward
                if done:
                    print(t + 1, total_rewards)
                    break
