import numpy as np
import tensorflow as tf
import random
from keras import Model
from keras.layers import Layer, Input, Activation
from keras.optimizers import Adam

from projet.dqn.dqnV3 import DQN
from projet.nDQN.noisy_dense import NoisyModel

class NDQN(DQN):
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000,
                 epsilon_decay=0.999, epsilon_min=0.10, gamma=0.89, trials=5000, batch_size=64, lr=0.001):
        super().__init__(env, model_factory, input_shape, batch_states_process,
                         observation_process, reward_process, memory_size,
                         epsilon_decay, epsilon_min, gamma, trials, batch_size, lr)

    # def init_models(self):
    #     self.model = NoisyModel(
    #         self.input_shape, self.env.action_space.n)
    #     self.model.summary()
    #     self.target_model = NoisyModel(
    #         self.input_shape, self.env.action_space.n)
    #     self.model.compile(loss="mean_squared_error", optimizer=Adam(
    #         lr=self.learning_rate))
    #     self.target_model.compile(loss="mean_squared_error", optimizer=Adam(
    #         lr=self.learning_rate))
    #     self.target_update()

    # def action(self, state):
    #     self.epsilon *= self.epsilon_decay
    #     self.epsilon = max(self.epsilon_min, self.epsilon)
    #     if random.random() <= self.epsilon:
    #         action = self.env.action_space.sample()
    #     else:
    #         action = np.argmax(self.model.predict(state)[0])
    #     return action

    # def replay(self, reward):
    #     if len(self.memory) < self.batch_size:
    #         return
    #     samples = random.sample(self.memory, self.batch_size)
    #     states = []
    #     actions = []
    #     rewards = []
    #     next_states = []
    #     dones = []
    #     for sample in samples:
    #         state, action, reward, next_state, done = sample
    #         states.append(state)
    #         actions.append(action)
    #         rewards.append(reward)
    #         next_states.append(next_state)
    #         if done:
    #             dones.append(0)
    #         else:
    #             dones.append(1)
        
    #     states = self.batch_states_process(states)
    #     actions = np.array(actions)
    #     rewards = np.array(rewards)
    #     next_states = self.batch_states_process(np.array(next_states))
    #     dones = np.array(dones)

    #     Q_targets = self.target_model.predict_on_batch(next_states)
    #     Q_targets = np.max(Q_targets, axis=1).flatten()
    #     Q_targets *= self.gamma
    #     Q_targets *= dones
    #     targets = self.target_model.predict_on_batch(states)
    #     for _, (target, R, action, d, q_t) in enumerate(zip(targets, rewards, actions, dones, Q_targets)):
    #         target[action] = R + q_t
    #     # print(target)
    #     loss = self.model.train_on_batch(states, targets)
    #     self.target_update()
    #     return loss
