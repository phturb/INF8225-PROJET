import numpy as np
import tensorflow as tf
import random
from keras import Model
from keras.layers import Layer, Input, Activation
from keras.optimizers import Adam

class NoisyDense(Layer):
    def __init__(self, out_features, in_features, std_init=0.4, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.use_bias = use_bias
        self.training = True

        mu_range = 1 / np.sqrt(in_features)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.out_features))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(out_features, in_features), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(out_features, in_features), dtype='float32'),
                                        trainable=True)
        if self.use_bias:
            self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(out_features,), dtype='float32'),
                                        trainable=True)

            self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(out_features,), dtype='float32'),
                                        trainable=True)

        self.sample_noise()
        
    def get_config(self):
        config = {}
        config.update({"out_features": self.out_features, "in_features":self.in_features, "std_init":self.std_init, "use_bias": self.use_bias})
        return config


    def call(self, inputs):
        self.kernel = self.weight_mu + (self.weight_sigma * self.weights_eps if self.training else 0)
        if self.use_bias:
            self.bias = self.bias_mu + (self.bias_sigma * self.bias_eps if self.training else 0)
            return tf.matmul(inputs, self.kernel) + self.bias
        return tf.matmul(inputs, self.kernel)

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    # def reset_noise(self):
    #     mu_range = 1.0 / np.sqrt(self.in_features)
    #     self.weight_mu.assign(tf.random_uniform_initializer(-mu_range, mu_range))
    #     self.weight_sigma.assign(tf.constant_initializer(self.std_init / np.sqrt(self.in_features)))
    #     if self.use_bias:
    #         self.bias_mu.assign(tf.random_uniform_initializer(-mu_range, mu_range))
    #         self.bias_sigma.assign(tf.constant_initializer(self.std_init / np.sqrt(self.out_features)))

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weights_eps = tf.multiply(tf.expand_dims(epsilon_in, 1), epsilon_out)
        if self.use_bias:
            self.bias_eps = epsilon_out

class NoisyModel(Model):
    def __init__(self, input_shape, action_n):
        inputs = Input(shape=input_shape[0])
        self.x1 = NoisyDense(24, inputs.shape[1])(inputs)
        x = Activation('relu')(self.x1)
        self.x2 = NoisyDense(64, x.shape[1])(x)
        x = Activation('relu')(self.x2)
        self.x3 = NoisyDense(24, x.shape[1])(x)
        x = Activation('relu')(self.x3)
        self.x4 = NoisyDense(action_n, x.shape[1])(x)
        action = Activation('linear')(self.x4)

        super().__init__(inputs=inputs, outputs=action)

    def sample_noise(self):
        print(type(self.x1))
        self.x1.sample_noise()
        self.x2.sample_noise()
        self.x3.sample_noise()
        self.x4.sample_noise()