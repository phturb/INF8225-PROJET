import numpy as np
import gym

import keras

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

import tensorflow as tf
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Activation, Layer
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from collections import deque

TRAIN = True
TEST = False

class NoisyDense(Layer):
    def __init__(self, units, input_dim, std_init=0.5, use_bias=True):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
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

    def get_config(self):
        config = {}
        config.update({"units": self.units, "input_dim":self.input_dim, "std_init":self.std_init, "use_bias": self.use_bias})
        return config

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

def create_model_ndqn(input_shape, action_n, lr):
        inputs = Input(shape=input_shape[0])
        x = NoisyDense(24, inputs.shape[1])(inputs)
        x = Activation('relu')(x)
        x = NoisyDense(64, x.shape[1])(x)
        x = Activation('relu')(x)
        x = NoisyDense(24, x.shape[1])(x)
        x = Activation('relu')(x)
        x = NoisyDense(action_n, x.shape[1])(x)
        action = Activation('linear')(x)
        model = Model(inputs=inputs, outputs=action)
        return model

class ProcessMountainCar(Processor):
    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        return batch.reshape(-1, 2)


ENV_NAME = 'MountainCar-v0'  # 'MountainCar-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
lr = 0.005
model = create_model_ndqn(env.observation_space.shape, nb_actions, lr)
print(model.summary())


# tb_log_dir = 'logs/tmp'
# tb_callback = TensorBoard()
memory = SequentialMemory(limit=2000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1600,
               target_model_update=1e-2, processor=ProcessMountainCar(), custom_model_objects = {"NoisyDense":NoisyDense})
optimizer = Adam(lr=lr, clipnorm=1.0)
dqn.compile(optimizer, metrics=['mae'])
if TRAIN:
    dqn.fit(env, nb_steps=100000, visualize=True,
            verbose=2)
    dqn.save_weights('dqn_keras_rl_{}_weights.h5f'.format(
        ENV_NAME), overwrite=True)
if TEST:
    dqn.load_weights('dqn_keras_rl_{}_weights.h5f'.format(
        ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
