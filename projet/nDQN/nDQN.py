import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Layer, Input, Activation
from keras.optimizers import Adam

from projet.dqn.dqnV3 import DQN

class NDQN(DQN):
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000,
                 epsilon_decay=0.999, epsilon_min=0.10, gamma=0.89, trials=5000, batch_size=64, lr=0.001):
        super().__init__(env, create_model_ndqn, input_shape, batch_states_process,
                         observation_process, reward_process, memory_size,
                         epsilon_decay, epsilon_min, gamma, trials, batch_size, lr)

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

def create_model_ndqn(input_shape, action_n):
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