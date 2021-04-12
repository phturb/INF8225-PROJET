import gym
import numpy as np
import random
from PIL import Image
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

from projet.agent import Agent

from copy import deepcopy
from collections import deque


class NDQN(Agent):
    def __init__(self, env, model_factory, input_shape=None, batch_states_process=None, observation_process=None, reward_process=None, memory_size=10000):
        super().__init__(env, model_factory, input_shape, batch_states_process,
                         observation_process, reward_process, memory_size)
