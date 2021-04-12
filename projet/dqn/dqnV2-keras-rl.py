import numpy as np
import gym

import keras

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from collections import deque

TRAIN = True
TEST = False


def create_model(input_shape, action_n, lr):
    inputs = Input(shape=input_shape)
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
model = create_model(env.observation_space.shape, nb_actions, lr)
print(model.summary())


tb_log_dir = 'logs/tmp'
tb_callback = TensorBoard()
memory = SequentialMemory(limit=2000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=32,
               target_model_update=1e-2, processor=ProcessMountainCar())
optimizer = Adam(lr=lr, clipnorm=1.0)
dqn.compile(optimizer, metrics=['mae'])
if TRAIN:
    dqn.fit(env, nb_steps=5000, visualize=True,
            verbose=2, callbacks=[tb_callback])
    dqn.save_weights('dqn_keras_rl_{}_weights.h5f'.format(
        ENV_NAME), overwrite=True)
if TEST:
    dqn.load_weights('dqn_keras_rl_{}_weights.h5f'.format(
        ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
