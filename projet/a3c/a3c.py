import gym
import numpy as np
import random
from PIL import Image
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

from copy import deepcopy
from collections import deque

TEST = False


def create_mountain_cart_model(input_shape, action_n):
    inputs = Input(shape=input_shape[0])
    x = Dense(24)(inputs)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(action_n)(x)
    action = Activation('linear')(x)
    model = Model(inputs=inputs, outputs=action)
    return model


def batch_states_process_mountain_cart(states):
    return np.array(states).reshape(-1, 2)


def observation_process_mountain_cart(state):
    return state.reshape(1, 2)


def reward_process_mountain_cart(reward):
    return reward


def create_atari_model(input_shape, action_n):
    inputs = Input(shape=input_shape)
    x = Convolution2D(16, (8, 8), strides=(4, 4))(inputs)
    x = Activation('relu')(x)
    x = Convolution2D(32, (4, 4), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(action_n)(x)
    action = Activation('linear')(x)
    model = Model(inputs=inputs, outputs=action)
    return model


def batch_states_process_atari(states):
    return np.array(states).reshape(-1, 84, 84)


def observation_process_atari(state):
    INPUT_SHAPE = (84, 84)
    img = Image.fromarray(state)
    img = img.resize(INPUT_SHAPE).convert(
        'L')  # resize and convert to grayscale
    processed_state = np.array(img) / 255
    return processed_state.reshape(1, 84, 84)


def reward_process_atari(reward):
    return reward


class DQNv2():
    def __init__(self, env, model_factory, batch_states_process, observation_process, input_shape=None, reward_process=None,
                 epsilon_decay=0.999, epsilon_min=0.10, memory_size=10000, gamma=0.89, trials=5000, trial_size=250, batch_size=64, lr=0.001):
        self.env = env
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.10
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.trials = trials
        self.trial_size = max(trial_size, self.env.spec.max_episode_steps)
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model_trained = False
        self.model = None
        self.target_model = None
        self.create_model = model_factory
        self.batch_states_process = batch_states_process
        self.observation_process = observation_process
        self.reward_process = reward_process
        self.input_shape = input_shape
        if self.input_shape is None:
            self.input_shape = self.env.observation_space.shape
    # def _create_atari

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

    def execute_step(self, state):
        action = self.action(state)
        next_state, reward, done, info = self.env.step(action)
        next_state = self.observation_process(next_state)
        if self.reward_process is not None:
            reward = self.reward_process(reward)
        return state, action, reward, next_state, done, info

    def replay(self):
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
        for idx, (target, R, action) in enumerate(zip(targets, Rs, actions)):
            targets[idx][action] = R
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

    def train(self, model_name="success.model", min_trials=100, win_ticks=-110):
        if self.model is None or self.target_model is None:
            self.init_models()
        self.total_rewards_list = deque(maxlen=min_trials)
        for trial in range(self.trials):
            current_state = self.observation_process(self.env.reset())
            total_rewards = 0
            for step in range(self.trial_size):
                state, action, reward, next_state, done, info = self.execute_step(
                    current_state)
                # if(trial % 15 == 0):
                #     self.env.render()
                # print(trial, step)
                self.env.render()
                self.remember(state, action, reward, next_state, done)
                self.replay()
                current_state = next_state
                total_rewards += reward
                if done:
                    break
            self.total_rewards_list.append(total_rewards)
            if step >= self.env.spec.max_episode_steps - 1:
                print("Failed to complete in trial {}".format(
                    trial), total_rewards)
            else:
                print("Completed in {} trials".format(trial), total_rewards)
                # self.model.save(model_name)
            if trial % 100 == 0:
                self.model.save(model_name)
            if trial >= min_trials and sum(self.total_rewards_list)/len(self.total_rewards_list) >= win_ticks:
                print("Solved in {} trials".format(trial), total_rewards)
                break
        self.model.save(model_name)

    def test(self):
        for t in range(5):
            current_state = self.observation_process(self.env.reset())
            for steps in range(self.trial_size):
                _, _, reward, current_state, done, _ = self.execute_step(
                    current_state)
                self.env.render()
                if done:
                    print(t + 1, reward)
                    break


def main():
    env = gym.make("Pong-v0")
    # dqn_agent = DQNv2(env=env, model_factory=create_atari_model,
    #                   batch_states_process=batch_states_process_atari,
    #                   observation_process=observation_process_atari,
    #                   reward_process=reward_process_atari,
    #                   input_shape=(84, 84, 1))
    env = gym.make("MountainCar-v0")
    dqn_agent = DQNv2(env=env, model_factory=create_mountain_cart_model,
                      batch_states_process=batch_states_process_mountain_cart,
                      observation_process=observation_process_mountain_cart,
                      reward_process=reward_process_mountain_cart)
    if TEST:
        dqn_agent.load_model()
        dqn_agent.test()
    else:
        dqn_agent.train()


main()
