import numpy as np
import random
from PIL import Image
from keras import Model
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten


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
