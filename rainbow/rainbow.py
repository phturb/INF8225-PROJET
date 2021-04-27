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
from keras.losses import MeanSquaredError,CategoricalCrossentropy # KLDivergence #CategoricalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import History, CallbackList

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

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


    def __init__(self, capacity=10000, e=0.01, a=0.6, beta=0.4, beta_max=1., beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_max = beta_max
        self.beta_increment_per_sampling = beta_increment_per_sampling

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

        self.beta = np.min([self.beta_max, self.beta + self.beta_increment_per_sampling])

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
    def __init__(self, units, activation=None, sigma=0.5, mu_initializer=None, sigma_w_initializer=None, sigma_b_initializer=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma = sigma
        self.activation = activations.get(activation)
        self.mu_initializer = initializers.get(mu_initializer)
        self.sigma_w_initializer = initializers.get(sigma_w_initializer)
        self.sigma_b_initializer = initializers.get(sigma_b_initializer)

    def build(self, input_shape):
        self.reset_noise(input_shape[-1])
        mu_range = 1 / np.sqrt(input_shape[-1])
        self.mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range, seed=SEED)
        self.sigma_w_initializer = tf.constant_initializer(self.sigma / np.sqrt(input_shape[-1]))
        self.sigma_b_initializer = tf.constant_initializer(self.sigma / np.sqrt(self.units))
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.mu_initializer, trainable=True, name='w_mu', dtype='float32')
        self.w_sigma = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.sigma_w_initializer, trainable=True, name='w_sigma', dtype='float32')
        self.b_mu = self.add_weight(shape=(self.units,), initializer=self.mu_initializer, trainable=True, name='b_mu', dtype='float32')
        self.b_sigma = self.add_weight(shape=(self.units,), initializer=self.sigma_b_initializer, trainable=True, name='b_sigma', dtype='float32')

    def call(self, inputs, training=True):
        if training:
            self.kernel = tf.add(self.w_mu, tf.multiply(self.w_sigma, self.w_eps))
            outputs = tf.matmul(inputs, self.kernel)
            self.bias = tf.add(self.b_mu, tf.multiply(self.b_sigma, self.b_eps))
            outputs = tf.nn.bias_add(outputs, self.bias)
        else:
            self.kernel = self.w_mu
            outputs = tf.matmul(inputs, self.kernel)
            self.bias = self.b_mu
            outputs = tf.nn.bias_add(outputs, self.bias)            
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.multiply(tf.sign(noise), tf.sqrt(tf.abs(noise)))

    def reset_noise(self, input_shape = None):
        if input_shape is None:
            input_shape = self.input_shape[-1]
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
            "sigma_w_initializer": initializers.serialize(self.sigma_w_initializer),
            "sigma_b_initializer" : initializers.serialize(self.sigma_b_initializer)
        })
        return config

def atari_state_processor(state):
    INPUT_SHAPE = (84, 84)
    CROP_SHAPE = (0,35,160,195)
    img = Image.fromarray(state)
    img = img.crop(CROP_SHAPE).resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_state = np.array(img) / 255
    processed_state = processed_state.reshape(84, 84, 1)
    return processed_state

class Rainbow():
    def __init__(self,
                 env,
                 memory_capacity=50000, # memory_capacity=1000000,
                 n_stacked_states=3,
                 model_name="rainbow",
                 epsilon_min=0.01, # epsilon_min=0
                 epsilon_decay=0.000033,#0.00000396,
                 gamma=0.99,
                 adam_epsilon=0.00015,
                 lr=0.0000625,
                 tau=1,
                 is_atari=False,
                 dd_enabled=False,
                 dueling_enabled=False,
                 noisy_net_theta=0.5, noisy_net_enabled=False,
                 prioritization_w=0.5, prioritization_b_min=0.4, prioritization_b_max=1, prioritized_memory_enabled=False,
                 atoms=51, v_min=-20, v_max=20, categorical_enabled=False):
        self.env = env
        self.env.seed(SEED)
        self.memory = PrioritizedMemory(capacity=memory_capacity) if prioritized_memory_enabled else DefaultMemory(max_size=memory_capacity)
        self.model_name = model_name
        self.action_dim = env.action_space.n
        self.epsilon = 1.
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr

        self.is_atari = is_atari
        self.n_stacked_states = n_stacked_states
        if self.is_atari:
            self.state_dim = (n_stacked_states,) + (84, 84, 1)
        else:
            self.state_dim = env.observation_space.shape
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
        self.z = np.arange(v_min, v_max + self.z_delta/2, self.z_delta)

        if self.categorical_enabled:
            def modified_KL_Divergence(y_true, y_pred):
                y_pred = tf.convert_to_tensor(y_pred)
                y_true = tf.cast(y_true, y_pred.dtype)
                return -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-7), axis=-1)
            self.crit = modified_KL_Divergence # CategoricalCrossentropy() # KL...
        else:
            self.crit = MeanSquaredError()
        self.adam_epsilon = adam_epsilon
        self.opt = Adam(learning_rate=self.lr, epsilon=adam_epsilon)
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
        if self.tau < 1:
            q_target_weights = self.q_target_model.get_weights()
            for i in range(len(q_target_weights)):
                q_target_weights[i] = self.tau * q_weights[i] + \
                    (1 - self.tau) * q_target_weights[i]
            self.q_target_model.set_weights(q_target_weights)
        else:
            self.q_target_model.set_weights(q_weights)

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
        dense_layer = NoisyDense if self.noisy_net_enabled else Dense
        name="DQN"
        if self.noisy_net_enabled:
            print("Noisy Net")
            name+="_Noisy_Net"
        if self.is_atari:
            inputs = Input(input_shape)
            x = Convolution2D(32, (8,8) , strides=(4, 4),activation='relu')(inputs)
            x = Convolution2D(64, (4,4), strides=(2, 2), activation='relu')(x)
            x = Convolution2D(64, (3,3), strides=(1, 1), activation='relu')(x)
            x = Flatten()(x)
            x = dense_layer(512, activation='relu')(x)
        else:
            inputs = Input(input_shape)
            x = dense_layer(24, activation='relu')(inputs)
            x = dense_layer(64, activation='relu')(x)
            x = dense_layer(24, activation='relu')(x)

        if self.dueling_enabled and not self.categorical_enabled:
            # Dueling
            print("Dueling")
            name+="_Dueling"
            a = dense_layer(output_shape, activation='linear')(x)
            v = dense_layer(1, activation='linear')(x)
            def avg_duel(s):
                a, v = s
                return v + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,))([a,v])
        elif not self.dueling_enabled and self.categorical_enabled:
            # Categorical (Distributional)
            print("Categorical")
            name+="_Categorical"
            outputs = [dense_layer(self.atoms, name=f"categorical_dense_{i}")(x) for i in range(output_shape)]
            outputs = Concatenate()(outputs)
            outputs = Reshape(( output_shape, self.atoms))(outputs)
            action = Activation('softmax')(outputs)
        elif self.dueling_enabled and self.categorical_enabled:
             # Categorical (Distributional) + Dueling
            print("Categorical + Dueling")
            name+="_Categorical_Dueling"
            a = []
            for _ in range(output_shape):
                a.append(dense_layer(self.atoms)(x))
            v = dense_layer(self.atoms, name=f"categorical_dense")(x)
            a = Concatenate()(a)
            a = Reshape(( output_shape , self.atoms))(a)
            def avg_duel(s):
                a, v = s
                return K.expand_dims(v, 1) + (a - K.mean(a, axis=1, keepdims=True))
            action = Lambda(avg_duel, output_shape=(output_shape,self.atoms))([a, v])
            action = Activation('softmax')(action)
        else:    
            # Default DQN
            action = dense_layer(output_shape, activation='linear')(x)
        return Model(inputs=inputs, outputs=action, name=name)

    def forward(self, state, testing=False):
        if testing:
            return self.predict_action(state)
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_dim - 1)
        else:
            return self.predict_action(state)


    def predict_action(self, state):
        q = self.q_model.predict(np.array([state]))
        if self.categorical_enabled:
            q = q * self.z
            q = np.sum(q, axis=2)
        return np.argmax(q[0])

    def backward(self, batch_size):
        samples = self.memory.sample(batch_size)
        if samples is None:
            return None
        if self.prioritized_memory_enabled:
            states, actions, rewards, new_states, dones, idxs, weights = samples
        else:
            states, actions, rewards, new_states, dones = samples

        targets = self.q_model.predict_on_batch(states)
        if self.categorical_enabled:
            p = targets
            targets = targets * self.z
            targets = np.sum(targets, axis=2)

        if self.prioritized_memory_enabled:
            old_targets = targets.copy()
        if not self.dd_enabled:
            keep_actions = np.argmax(targets, axis=1)
            Q_target = self.q_target_model.predict_on_batch(new_states)
            p_ = Q_target
            Q_target = Q_target[range(batch_size), keep_actions]
        else:
            Q_target = self.q_target_model.predict_on_batch(new_states)
            if self.categorical_enabled:
                p_ = Q_target
                keep_actions = p_ * self.z
                keep_actions = np.sum(keep_actions, axis=2)
                keep_actions = np.argmax(keep_actions, axis=1)
            else:
                keep_actions = np.argmax(Q_target, axis=1)
            Q_target = Q_target[range(batch_size), keep_actions]
        if self.categorical_enabled:
            m = np.zeros((batch_size, self.action_dim, self.atoms))
            for i in range(batch_size):
                for j in range(self.atoms):
                    a = actions[i]
                    b_a = keep_actions[i]
                    d = 0 if dones[i] else 1
                    tz = max(self.v_min ,min(self.v_max, rewards[i] + d * self.gamma * self.z[j]))
                    bj = (tz - self.v_min) / self.z_delta
                    l, u = np.floor(bj), np.ceil(bj)
                    if not dones[i]:
                        m[i][a][int(l)] += p_[i][b_a][j] * (u - bj)
                        m[i][a][int(u)] += p_[i][b_a][j] * (bj - l)
                    else:
                        m[i][a][int(l)] += (u - bj)
                        m[i][a][int(u)] += (bj - l)
                    # for o_a in range(self.action_dim):
                    #     if o_a != a:
                    #         m[i][o_a][j] = p[i][o_a][j]
            targets = m
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
                new_targets = p * self.z
                new_targets = np.sum(new_targets, axis=2)
            else:
                new_targets = targets
            
            errors = np.abs(new_targets[indices, actions] -
                            old_targets[indices, actions])
            for i in range(batch_size):
                self.memory.update(idxs[i], errors[i])
        loss = self.q_model.train_on_batch(states, targets)
        if self.noisy_net_enabled:
            for layer in self.q_model.layers:
                if hasattr(layer, "reset_noise"):
                    layer.reset_noise()
        return loss

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

    def train(self, max_trials=500, batch_size=64, warmup=80000, model_update_delay=1, render=False, n_step=1, callbacks=None, avg_result_exit=480, avg_list_lenght=50):
        assert n_step > 0

        callbacks = [] if not callbacks else callbacks[:]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self.q_model)
        else:
            callbacks._set_model(self.q_model)

        # Fast exit on last 50 values
        total_rewards_history = deque(maxlen=avg_list_lenght)

        callbacks.on_train_begin()

        # Multistep initializer
        self.n_step = n_step
        if self.n_step > 1:
            self.multistep_buffer = []

        total_trials_steps = 0
        trial = 0
        while trial < max_trials:
            episode_start_time = time.time()
            callbacks.on_epoch_begin(trial)
            done = False
            trial_total_reward = 0
            trial_steps = 0

            current_state = self.env.reset()
            if self.is_atari:
                current_state = atari_state_processor(current_state)
                stacked_state = []
                for _ in range(self.n_stacked_states):
                    stacked_state.append(current_state)
                current_state = stacked_state
                
            while not done:
                start_time = time.time()
                callbacks.on_batch_begin(trial_steps)                   

                if render:
                    self.env.render()

                action = self.forward(current_state)
                next_state, reward, done, info = self.env.step(action)
                if self.is_atari:
                    next_state = atari_state_processor(next_state)
                    next_state = np.append(stacked_state[1:], [next_state], axis=0)
                self.remember(current_state, action, reward, next_state , done)
                metrics = None
                if warmup <= total_trials_steps:
                    metrics = self.backward(batch_size)

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
                callbacks.on_batch_end(total_trials_steps, step_logs)
                trial_total_reward += reward
                trial_steps += 1
                total_trials_steps += 1
            episode_end_time = time.time()

            total_rewards_history.append(trial_total_reward)
            episode_logs = {
                'episode_reward': trial_total_reward,
                'nb_episode_steps': trial_steps,
                'nb_steps': total_trials_steps,
                'episode_time': (episode_end_time - episode_start_time)
            }
            self.multistep_reset()

            
            callbacks.on_epoch_end(trial, episode_logs)

            if len(total_rewards_history) >= avg_list_lenght:
                avg = sum(total_rewards_history)/len(total_rewards_history)
                if avg >= avg_result_exit:
                    print(f'Model has trained over the average : {avg_result_exit}')
                    break

            print(f"Trial {trial} complete with reward : {trial_total_reward} in {episode_logs['episode_time']}ms")
            trial += 1
        callbacks.on_train_end()
        return history

    def test(self, max_trials=5, render=False):
        self.n_step = 1
        for trial in range(max_trials):
            done = False
            trial_total_reward = 0
            current_state = self.env.reset()
            while not done:
                if render:
                    self.env.render()
                action = self.forward(current_state, testing=True)
                next_state, reward, done, info = self.env.step(action)
                trial_total_reward += reward
                current_state = next_state
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
