import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Comment this line to use GPU instead of CPU
import numpy as np
import random
import time
import tensorflow as tf
import keras.backend as K

from PIL import Image
from copy import deepcopy
from collections import deque

from keras import activations , initializers
from keras.layers import Input, Dense, Layer, Activation, Lambda, Convolution2D, Flatten, Concatenate,Reshape 
from keras.models import Model, load_model
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import History, CallbackList


class DefaultMemory():
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def sample(self, sample_size):
        if len(self.memory) < sample_size:
            return None
        samples = random.sample(self.memory, sample_size)
        states, actions, rewards, new_states, dones = map(
            np.asarray, zip(*samples))
        return states, actions, rewards, new_states, dones

    def append(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def size(self):
        return len(self.memory)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity=10000):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def append(self, state, action, reward, new_state, done):
        p = np.max(self.tree.tree[-self.tree.capacity:])
        p = p if p != 0 else 1
        self.tree.add(p, (state, action, reward, new_state, done))

    def sample(self, n):
        if self.size() < n:
            return None

        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states, actions, rewards, new_states, dones = map(
            np.asarray, zip(*batch))

        return states, actions, rewards, new_states, dones, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class NoisyDense(Layer):
    def __init__(self, units, activation=None, sigma=0.5, mu_initializer=None, sigma_initializer=None, w_eps=None, b_eps=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma = sigma
        self.activation = activations.get(activation)
        self.mu_initializer = initializers.get(mu_initializer)
        self.sigma_initializer = initializers.get(sigma_initializer)
        self.w_eps = w_eps
        self.b_eps = b_eps

    def build(self, input_shape):
        if self.w_eps is None or self.b_eps is None:
            self.reset_noise(input_shape[-1])
        mu_range = 1 / np.sqrt(input_shape[-1])
        self.mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        self.sigma_initializer = tf.constant_initializer(self.sigma / np.sqrt(self.units))
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.mu_initializer, trainable=True, name='w_mu')
        self.w_sigma = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.sigma_initializer, trainable=True, name='w_sigma')
        self.b_mu = self.add_weight(shape=(self.units,), initializer=self.mu_initializer, trainable=True, name='b_mu')
        self.b_sigma = self.add_weight(shape=(self.units,), initializer=self.sigma_initializer, trainable=True, name='b_sigma')

    def call(self, inputs):
        self.kernel = self.w_mu + self.w_sigma * self.w_eps
        self.bias = self.b_mu + self.b_sigma * self.b_eps
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.w_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.b_eps = eps_out

    def get_config(self):
        config = super(NoisyDense, self).get_config()
        config.update({
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "sigma": self.sigma,
            "mu_initializer": initializers.serialize(self.mu_initializer),
            "sigma_initializer": initializers.serialize(self.sigma_initializer),
            "b_eps": self.b_eps.numpy(),
            "w_eps": self.w_eps.numpy()
        })
        return config

def atari_state_processor(state):
    INPUT_SHAPE = (84, 84)
    CROP_SHAPE = (0,35,160,195)
    img = Image.fromarray(state)
    img = img.crop(CROP_SHAPE).resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_state = np.array(img) / 255
    processed_state = processed_state.reshape(1 ,84, 84, 1)
    return processed_state

class Rainbow():
    def __init__(self,
                 env,
                 model_name="rainbow",
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 gamma=0.99,
                 lr=0.0005,  # lr=0.0000625,
                 tau=1,
                 is_atari=False,
                 dd_enabled=False,
                 dueling_enabled=False,
                 noisy_net_theta=0.5, noisy_net_enabled=False,
                 prioritization_w=0.5, prioritized_memory_enabled=False,
                 atoms=51, v_min=-10, v_max=10, categorical_enabled=False):
        self.env = env
        self.memory = PrioritizedMemory() if prioritized_memory_enabled else DefaultMemory()
        self.model_name = model_name
        self.action_dim = env.action_space.n
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr

        self.is_atari = is_atari
        if self.is_atari:
            self.state_dim = (84, 84, 1)
        else:
            self.state_dim = env.observation_space.shape[0]

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
        assert atoms >= 1
        assert v_min < v_max
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_delta = (v_max - v_min) / (atoms - 1)
        self.z = np.array([ (v_min + i*self.z_delta) for i in range(atoms)  ])

        if self.categorical_enabled:
            self.crit = CategoricalCrossentropy() # KL...
        else:
            self.crit = MeanSquaredError()
        self.opt = Adam(learning_rate=self.lr)
        self.init_models()

    def init_models(self):
        self.q_model = self.create_model(self.state_dim, self.action_dim)
        self.q_model.compile(loss=self.crit, optimizer=self.opt)
        self.q_model.summary()
        self.q_target_model = self.create_model(
            self.state_dim, self.action_dim)
        self.update_model()

    def update_model(self):
        q_weights = self.q_model.get_weights()
        q_target_weights = self.q_target_model.get_weights()
        for i in range(len(q_target_weights)):
            q_target_weights[i] = self.tau * q_weights[i] + \
                (1 - self.tau) * q_target_weights[i]
        self.q_target_model.set_weights(q_target_weights)

    def save_models(self):
        base_path = f"./models/{self.model_name}/base"
        target_path = f"./models/{self.model_name}/target"
        self.q_model.save(base_path, overwrite=True)
        self.q_target_model.save(target_path, overwrite=True)

    def load_models(self):
        base_path = f"./models/{self.model_name}/base"
        target_path = f"./models/{self.model_name}/target"
        self.q_model = load_model(base_path, custom_objects={"NoisyDense": NoisyDense})
        self.q_target_model = load_model(target_path, custom_objects={"NoisyDense": NoisyDense})

    def create_model(self, input_shape, output_shape):
        
        if self.is_atari:
            inputs = Input(input_shape)
            x = Convolution2D(16, (8, 8), strides=(4, 4))(inputs)
            x = Activation('relu')(x)
            x = Convolution2D(32, (4, 4), strides=(2, 2))(x)
            x = Activation('relu')(x)
            x = Flatten()(x)
            if self.noisy_net_enabled:
                x = NoisyDense(256, x.shape[1])(x)
                x = Activation('relu')(x)
            else:
                x = Dense(256, activation='relu')(x)
        else:
            inputs = Input((input_shape,))
            if self.noisy_net_enabled:
                x = NoisyDense(24, activation='relu')(inputs)
                x = NoisyDense(64, activation='relu')(x)
                x = NoisyDense(24, activation='relu')(x)
            else:
                x = Dense(24, activation='relu')(inputs)
                x = Dense(64, activation='relu')(x)
                x = Dense(24, activation='relu')(x)

        if self.noisy_net_enabled and not self.dueling_enabled and not self.categorical_enabled:
            # Noisy Net
            print("Noisy Net")
            action = NoisyDense(output_shape, activation='linear')(x)
        elif not self.noisy_net_enabled and self.dueling_enabled and not self.categorical_enabled:
            # Dueling
            print("Dueling")
            a = Dense(output_shape + 1, activation='linear')(x)
            v = Dense(1, activation='linear')(x)
            def avg_duel(s):
                a, v = s
                return v + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,))([a,v])
        elif self.noisy_net_enabled and self.dueling_enabled and not self.categorical_enabled:
            # Dueling + Noisy Net
            print(" Dueling + Noisy Net")
            a = NoisyDense(output_shape + 1, activation='linear')(x)
            v = Dense(1, activation='linear')(x)
            def avg_duel(s):
                a , v = s
                return v + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,))([a,v])
        elif not self.noisy_net_enabled and not self.dueling_enabled and self.categorical_enabled:
            # Categorical (Distributional)
            print("Categorical")
            outputs = [Dense(self.atoms, name=f"categorical_dense_{i}")(x) for i in range(output_shape)]
            outputs = Concatenate()(outputs)
            outputs = Reshape(( output_shape, self.atoms))(outputs)
            action = Activation('softmax')(outputs)
        elif self.noisy_net_enabled and not self.dueling_enabled and self.categorical_enabled:
            # Categorical (Distributional) + Noisy Net
            print("Categorical + Noisy Net")
            outputs = []
            for _ in range(output_shape):
                outputs.append(NoisyDense(self.atoms)(x))
            outputs = Concatenate()(outputs)
            outputs = Reshape(( output_shape, self.atoms))(outputs)
            action = Activation('softmax')(outputs)
        elif not self.noisy_net_enabled and self.dueling_enabled and self.categorical_enabled:
             # Categorical (Distributional) + Dueling
            print("Categorical + Dueling")
            outputs = []
            for _ in range(output_shape):
                outputs.append(Dense(self.atoms)(x))
            v = Dense(self.atoms, name=f"categorical_dense")(x)
            outputs = Concatenate()(outputs)
            outputs = Reshape(( output_shape , self.atoms))(outputs)
            def avg_duel(s):
                a, v = s
                return K.expand_dims(v, 1) + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,self.atoms))([outputs, v])
            action = Activation('softmax')(action)
            
            # action = outputs
        elif self.noisy_net_enabled and self.dueling_enabled and self.categorical_enabled:
             # Categorical (Distributional) + Noisy Net + Dueling
            print("Categorical + Noisy Net + Dueling")
            outputs = []
            for _ in range(output_shape):
                outputs.append(NoisyDense(self.atoms)(x))
            v = NoisyDense(self.atoms, name=f"categorical_dense")(x)
            outputs = Concatenate()(outputs)
            outputs = Reshape(( output_shape , self.atoms))(outputs)
            def avg_duel(s):
                a, v = s
                return K.expand_dims(v, 1) + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,self.atoms))([outputs, v])
            action = Activation('softmax')(action)
        else:    
            # Default DQN
            action = Dense(output_shape, activation='linear')(x)
        return Model(inputs=inputs, outputs=action)

    def action(self, state, testing=False):
        if testing:
            return self.predict_action(state)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.predict_action(state)


    def predict_action(self, state):
        if self.is_atari:   
            q = self.q_model.predict(state)
        else:
            q = self.q_model.predict(np.reshape(state, [1, self.state_dim]))
        if self.categorical_enabled:
            q = q * self.z
            return np.argmax(np.sum(q[0], axis=1))
        return np.argmax(q[0])

    def replay(self, batch_size):
        samples = self.memory.sample(batch_size)
        if samples is None:
            return None
        if self.prioritized_memory_enabled:
            states, actions, rewards, new_states, dones, idxs, weights = samples
        else:
            states, actions, rewards, new_states, dones = samples

        if self.is_atari:
            new_states = np.reshape(new_states , (batch_size, 84, 84, 1))      
            states = np.reshape(states , (batch_size, 84, 84, 1))  

        targets = self.q_model.predict_on_batch(states)
        if self.categorical_enabled:
            p = targets
            targets = targets * self.z
            targets = np.sum(targets, axis=2)

        if self.prioritized_memory_enabled:
            old_targets = targets.copy()
        if self.dd_enabled:
            keep_actions = np.argmax(targets, axis=1)
            Q_target = self.q_target_model.predict_on_batch(new_states)
            Q_target = Q_target[range(batch_size), keep_actions]
        else:
            Q_target = self.q_target_model.predict_on_batch(new_states)
            if self.categorical_enabled:
                keep_actions = Q_target * self.z
                keep_actions = np.sum(keep_actions, axis=2)
                keep_actions = np.argmax(keep_actions, axis=1)
            else:
                keep_actions = np.argmax(Q_target, axis=1)
            Q_target = Q_target[range(batch_size), keep_actions]
        if self.categorical_enabled:
            m = np.zeros((batch_size, self.action_dim, self.atoms))
            for i in range(batch_size):
                for j in range(self.atoms):
                    d = 0 if dones[i] else 1
                    tz = max(self.v_min ,min(self.v_max, rewards[i] + d * self.gamma * self.z[j]))
                    bj = (tz - self.v_min) / self.z_delta
                    l, u = np.floor(bj), np.ceil(bj)
                    m[i][keep_actions[i]][int(l)] += Q_target[i][j] * (u - bj)
                    m[i][keep_actions[i]][int(u)] += Q_target[i][j] * (bj - l)
            targets = m * p
        else:
            Q_target = Q_target * self.gamma
            for i, (target, r, action, q_t, d) in enumerate(zip(targets, rewards, actions, Q_target, dones)):
                if d:
                    target[action] = r
                else:
                    target[action] = r + q_t

        if self.prioritized_memory_enabled:
            indices = np.arange(batch_size)
            
            if self.categorical_enabled:
                new_targets = targets * self.z
                new_targets = np.sum(new_targets, axis=2)
            else:
                new_targets = targets
            
            errors = np.abs(new_targets[indices, actions] -
                            old_targets[indices, actions])
            for i in range(batch_size):
                self.memory.update(idxs[i], errors[i])

        return self.q_model.train_on_batch(states, targets)

    def remember(self, current_state, action, reward, next_state, done):
        if self.n_step > 1:
            # Multistep
            self.multistep_buffer.append([current_state, action, reward, done])
            if len(self.multistep_buffer) < self.n_step:
                return
            reward = sum([self.multistep_buffer[i][2]*(self.gamma**i)
                         for i in range(self.n_step)])
            current_state, action, _, _ = self.multistep_buffer.pop(0)

        self.memory.append(current_state, action, reward, next_state, done)

    def multistep_reset(self):
        """ Multistep """
        if self.n_step <= 1:
            return
        while len(self.multistep_buffer) > 0:
            reward = sum([self.multistep_buffer[i][2]*(self.gamma**i)
                         for i in range(len(self.multistep_buffer))])
            state, action, _, _ = self.multistep_buffer.pop(0)
            self.memory.append(state, action, reward, state, True)

    def train(self, max_trials=500, batch_size=32, warmup=0, model_update_delay=1, render=False, n_step=1, callbacks=None):
        assert n_step > 0

        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self.q_model)
        else:
            callbacks._set_model(self.q_model)

        callbacks.on_train_begin()

        self.n_step = n_step
        if self.n_step > 1:
            self.multistep_buffer = []
        total_trials_steps = 0
        for trial in range(max_trials):
            episode_start_time = time.time()
            callbacks.on_epoch_begin(trial)
            done = False
            trial_total_reward = 0
            trial_steps = 0

            current_state = deepcopy(self.env.reset())
            if self.is_atari:
                current_state = atari_state_processor(current_state)
            while not done:
                start_time = time.time()
                callbacks.on_batch_begin(trial_steps)                   

                if render:
                    self.env.render()

                action = self.action(current_state)
                next_state, reward, done, info = self.env.step(action)
                if self.is_atari:
                    next_state = atari_state_processor(next_state)
                self.remember(current_state, action, reward, next_state, done)

                metrics = None
                if warmup <= total_trials_steps:
                    metrics = self.replay(batch_size)

                if total_trials_steps % model_update_delay == 0:
                    self.update_model()

                current_state = next_state
                end_time = time.time()
                step_logs = {
                    'action': action,
                    'observation': current_state,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': trial,
                    'time': (end_time - start_time)
                }
                callbacks.on_batch_end(trial_steps, step_logs)
                trial_total_reward += reward
                trial_steps += 1

            episode_end_time = time.time()

            episode_logs = {
                'episode_reward': trial_total_reward,
                'nb_episode_steps': trial_steps,
                'nb_steps': total_trials_steps,
                'episode_time': (episode_end_time - episode_start_time)
            }

            self.multistep_reset()
            
            callbacks.on_epoch_end(trial, episode_logs)
            total_trials_steps += trial_steps
            print(f"Trial {trial} complete with reward : {trial_total_reward} in {episode_logs['episode_time']}ms")
        callbacks.on_train_end()
        return history

    def test(self, max_trials=5, render=False):
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


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v1")
    # rain = Rainbow(env, memory=DefaultMemory(), dd_enabled=False, dueling_enabled=True, noisy_net_enabled=False)
    # env = gym.make("Pong-v0")
    rain = Rainbow(env, dd_enabled=True, dueling_enabled=True, noisy_net_enabled=False, prioritized_memory_enabled=True, is_atari=False)

    # callbacks = [TensorBoard(log_dir="./logs/rainbow/dueling", histogram_freq=1)]
    callbacks = None
    # n_step > 1 activate multistep
    rain.train(render=True, n_step=4, callbacks=callbacks)
