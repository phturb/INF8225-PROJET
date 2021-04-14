import numpy as np
import random
from PIL import Image
from keras import Model
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten


def create_cart_pole_model(input_shape, action_n):
    inputs = Input(shape=4)
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

def batch_states_process_cart_pole(states):
    return np.array(states).reshape(-1, 4)


def observation_process_cart_pole(state):
    return state.reshape(1, 4)


def reward_process_cart_pole(reward):
    return reward
