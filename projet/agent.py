import gym
from collections import deque
from copy import deepcopy
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import Callback, CallbackList, History
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
        """Selection d'action"""
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
        # if self.reward_process is not None:
        #     reward = self.reward_process(reward)
        return next_state, reward, done, info

    def get_model(self):
        raise NotImplementedError()

    def train(self, trials=5000, trial_size=None, threshold=None, model_name="success.model", min_trials=100, render=False, callbacks=None,
              solve_condition=lambda total_rewards_list, trial, min_trials, threshold: trial >= 100 and sum(total_rewards_list)/len(total_rewards_list) >= threshold):
        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self.get_model())
        else:
            callbacks._set_model(self.get_model())
        callbacks.on_train_begin()

        self.total_rewards_list = deque(maxlen=min_trials)
        if trial_size is None:
            trial_size = self.env.spec.max_episode_steps
        if threshold is None:
            threshold = self.env.spec.reward_threshold
        trial = 0
        step = 0
        while trial <= trials:
            total_rewards = 0
            current_step = 0
            callbacks.on_epoch_begin(trial)
            done = False
            current_state = self.observation_process(
                deepcopy(self.env.reset()))
            while not done:
                callbacks.on_batch_begin(current_step)

                if render:
                    self.env.render()

                # callbacks.on_action_begin(action)

                action = self.action(current_state)
                next_state, reward, done, _ = self.execute_action(action)

                # callbacks.on_action_end(action)

                self.remember(current_state, action, reward, next_state, done)
                metrics = self.replay(reward)
                current_state = next_state
                step_logs = {
                    'action': action,
                    'observation': current_state,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': trial,
                }
                callbacks.on_batch_end(current_step, step_logs)
                total_rewards += reward
                step+=1
                current_step+=1
                if done:
                    episode_logs = {
                        'episode_reward': total_rewards,
                        'nb_episode_steps': current_step,
                        'nb_steps': step,
                    }

                    break
            callbacks.on_epoch_end(trial, episode_logs)
            self.total_rewards_list.append(total_rewards)
            trial+=1
            if current_step >= self.env.spec.max_episode_steps - 1:
                print("Failed to complete in trial {}".format(
                    trial), total_rewards)
            else:
                print("Completed in {} trials".format(trial), total_rewards)
            if trial % 100 == 0:
                self.save_models(model_name)
            if solve_condition(self.total_rewards_list, trial, min_trials, threshold):
                print("Solved in {} trials".format(trial), total_rewards)
                break
        callbacks.on_train_end()
        self.save_models(model_name)
        return history
        

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



class MultistepAgent(Agent):
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000,
                n_step=4, gamma=.99):
        assert n_step > 0
        self.n_step = n_step
        self.gamma = gamma
        self.multistep_buffer = []
        super().__init__(env, model_factory, input_shape, batch_states_process,
                         observation_process, reward_process, memory_size)

    def remember(self, state, action, reward, new_state, done):
        self.multistep_buffer.push([state, action, reward, new_state, done])
        if len(self.multistep_buffer) < self.n_step:
            return
        
        reward = sum([self.multistep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.multistep_buffer.pop(0)

        self.memory.append([state, action, reward, new_state, done])

    def multistep_reset(self):
        while len(self.multistep_buffer) > 0:
            reward = sum([self.multistep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.multistep_buffer.pop(0)

            self.memory.push([state, action, reward, None, done])


    def execute_action(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.observation_process(deepcopy(next_state))
        if self.reward_process is not None:
            reward = self.reward_process(reward)
        return next_state, reward, done, info

    def train(self, trials=5000, trial_size=None, threshold=None, model_name="success.model", min_trials=100, render=False, callbacks=None,
              solve_condition=lambda total_rewards_list, trial, min_trials, threshold: trial >= 100 and sum(total_rewards_list)/len(total_rewards_list) >= threshold):
        
        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self.get_model())
        else:
            callbacks._set_model(self.get_model())
        callbacks.on_train_begin()

        self.total_rewards_list = deque(maxlen=min_trials)
        if trial_size is None:
            trial_size = self.env.spec.max_episode_steps
        if threshold is None:
            threshold = self.env.spec.reward_threshold
        trial = 0
        step = 0
        while trial <= trials:
            self.multistep_buffer = []
            total_rewards = 0
            current_step = 0
            callbacks.on_epoch_begin(trial)
            done = False
            current_state = self.observation_process(
                deepcopy(self.env.reset()))
            while not done:
            # for step in range(trial_size):
                callbacks.on_batch_begin(current_step)

                if render:
                    self.env.render()

                # callbacks.on_action_begin(action)

                action = self.action(current_state)
                next_state, reward, done, _ = self.execute_action(action)

                # callbacks.on_action_end(action)

                self.remember(current_state, action, reward, next_state, done)
                metrics = self.replay(reward)
                current_state = next_state
                step_logs = {
                    'action': action,
                    'observation': current_state,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': trial,
                }
                callbacks.on_batch_end(current_step, step_logs)
                total_rewards += reward
                step+=1
                current_step+=1
                if done:
                    self.multistep_reset()
                    episode_logs = {
                        'episode_reward': total_rewards,
                        'nb_episode_steps': current_step,
                        'nb_steps': step,
                    }
                    break
            callbacks.on_epoch_end(trial, episode_logs)
            self.total_rewards_list.append(total_rewards)
            trial+=1
            if current_step >= self.env.spec.max_episode_steps - 1:
                print("Failed to complete in trial {}".format(
                    trial), total_rewards)
            else:
                print("Completed in {} trials".format(trial), total_rewards)
            if trial % 100 == 0:
                self.save_models(model_name)
            if solve_condition(self.total_rewards_list, trial, min_trials, threshold):
                print("Solved in {} trials".format(trial), total_rewards)
                break
        callbacks.on_train_end()
        self.save_models(model_name)
        return history

    def test(self, trials=5, trial_size=None, render=False):
        if trial_size is None:
            trial_size = self.env.spec.max_episode_steps
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