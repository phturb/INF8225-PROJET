import numpy as np
import random
import tensorflow as tf

from copy import deepcopy
from collections import deque

from keras.layers import Input, Dense, Layer, Activation
from keras.models import Model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

class DefaultMemory():
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
    
    def sample(self, sample_size):
        if len(self.memory) < sample_size:
            return None
        samples = random.sample(self.memory, sample_size)
        states, actions, rewards, new_states, dones = map(np.asarray, zip(*samples))
        return states, actions, rewards, new_states, dones

    def append(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def size(self):
        return len(self.memory)

class NoisyDense(Layer):
    def __init__(self, units, input_dim, std_init=0.5, use_bias=True):
        super().__init__()
        self.units = units
        self.std_init = std_init
        self.use_bias = use_bias
        self.reset_noise(input_dim)

        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, units), dtype='float32'),
                                        trainable=True)
        if self.use_bias:
            self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(units,), dtype='float32'),
                                        trainable=True)

            self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(units,), dtype='float32'),
                                        trainable=True)

    def call(self, inputs):
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        if self.use_bias:
            self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        if self.use_bias:
            self.bias_eps = eps_out

class Rainbow():
    def __init__(self,
            env,
            memory=None,
            epsilon_min=0.1,
            epsilon_decay=0.995,
            gamma=0.99, 
            lr=0.0005, # lr=0.0000625,
            tau=0.8,
            dd_enabled=False,
            dueling_enabled=False, 
            noisy_net_theta=0.5, noisy_net_enabled=False,
            prioritization_w=0.5, prioritized_memory_enabled=False,
            atoms=51, v_min=-10, v_max=10, categorical_enabled=False):
        self.env = env
        self.memory = memory
        if self.memory is None:
            self.memory = DefaultMemory()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr

        self.tau = tau
        # DDQn
        self.dd_enabled = dd_enabled
        # Dueling Network
        self.dueling_enabled = dueling_enabled
        # Noisy Net
        self.noisy_net_enabled = noisy_net_enabled
        self.noisy_net_theta = noisy_net_theta
        # Prioritized
        self.prioritized_memory_enabled = prioritized_memory_enabled
        self.prioritization_w = prioritization_w
        # Categorical
        self.categorical_enabled = categorical_enabled
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        self.crit = MeanSquaredError()
        self.opt = Adam(learning_rate=self.lr)
        self.init_models()

    def init_models(self):
        self.q_model = self.create_model(self.state_dim, self.action_dim)
        self.q_model.compile(loss=self.crit, optimizer=self.opt)
        self.q_target_model = self.create_model(self.state_dim, self.action_dim)
        self.update_model()

    def update_model(self):
        q_weights = self.q_model.get_weights()
        if self.dd_enabled:
            q_target_weights = self.q_target_model.get_weights()
            for i in range(len(target_weights)):
                q_target_weights[i] = self.tau * q_weights[i] + \
                    (1 - self.tau) * q_target_weights[i]
        else:
            q_target_weights = q_weights
        self.q_target_model.set_weights(q_target_weights)

    def create_model(self, input_shape, output_shape):
        inputs = Input((input_shape,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(24, activation='relu')(x)
        if self.noisy_net_enabled:
            action = NoisyDense(output_shape, x.shape[1])
            action = Activation('linear')
        else:    
            action = Dense(output_shape, activation='linear')(x)
        return Model(inputs=inputs, outputs=action)

    def action(self, state, testing=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.predict_action(state)

    def predict_action(self, state):
        q = self.q_model.predict(np.reshape(state, [1, self.state_dim]))
        return np.argmax(q[0])

    def replay(self, batch_size):
        samples = self.memory.sample(batch_size)
        if samples is None:
            return None
        states, actions, rewards, new_states, dones = samples
        targets = self.q_model.predict_on_batch(states)
        Q_target = self.q_target_model.predict_on_batch(new_states)
        Q_target = np.max(Q_target, axis=1).flatten()
        Q_target = Q_target * self.gamma
        
        for i, (target, r, action, q_t, d) in enumerate(zip(targets, rewards, actions, Q_target, dones)):
            if d:
                target[action] = r
            else:
                target[action] = r + q_t
        return self.q_model.train_on_batch(states, targets)

    def remember(self, current_state, action, reward, next_state, done):
        if self.n_step > 1:
             # Multistep
            self.multistep_buffer.append([current_state, action, reward, done])
            if len(self.multistep_buffer) < self.n_step:
                return
            reward = sum([self.multistep_buffer[i][2]*(self.gamma**i) for i in range(self.n_step)])
            current_state, action, _, _ = self.multistep_buffer.pop(0)
    
        self.memory.append(current_state, action, reward, next_state, done)
    
    def multistep_reset(self):
        """ Multistep """
        if self.n_step <= 1:
            return
        while len(self.multistep_buffer) > 0:
            reward = sum([self.multistep_buffer[i][2]*(self.gamma**i) for i in range(len(self.multistep_buffer))])
            state, action, _, _ = self.multistep_buffer.pop(0)
            self.memory.append(state, action, reward, state, True)


    def train(self, max_trials=500, batch_size=32, warmup=0, model_update_delay=1, render=False, n_step=1):
        self.n_step = n_step
        if self.n_step > 1:
            self.multistep_buffer = []
        total_trials_steps = 0
        for trial in range(max_trials):
            done = False
            trial_total_reward = 0
            trial_steps = 0

            current_state = deepcopy(self.env.reset())
            while not done:

                if render:
                    self.env.render()

                action = self.action(current_state)
                next_state, reward, done, info = self.env.step(action)
                self.remember(current_state, action, reward, next_state, done)

                if warmup <= total_trials_steps:
                    self.replay(batch_size)

                if total_trials_steps % model_update_delay == 0:
                    self.update_model()

                current_state = next_state
                trial_total_reward += reward
                trial_steps += 1
    
            self.multistep_reset()

            total_trials_steps+=trial_steps
            print(f"Trial {trial} complete with reward : {trial_total_reward}")

    def test(self, trials=5, render=False):
        self.n_step = 1
        for trial in range(max_trials):
            done = False
            trial_total_reward = 0
            current_state = deepcopy(self.env.reset())
            while not done:
                if render:
                    self.env.render()
                action = self.action(current_state, testing=True)
                next_state, reward, done, info = self.env.step(action)
                trial_total_reward += reward
            print(f"Trial {trial} complete with reward : {trial_total_reward}")


import gym

env = gym.make("CartPole-v1")
rain = Rainbow(env, memory=DefaultMemory(), noisy_net_enabled=False)


# n_step > 1 activate multistep
rain.train(render=True, n_step=1)