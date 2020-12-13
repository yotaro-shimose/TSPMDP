from collections import defaultdict
from typing import Callable, List

import numpy as np
import tensorflow as tf
import tree
from tspmdp.dqn.server import Server
from tspmdp.env import TSPMDP
from tspmdp.logger import TFLogger

INFINITY = 1e+9


def map_dict(func, arg_dict: dict):
    return {key: func(arg) for key, arg in arg_dict.items()}


class Actor:
    def __init__(
        self,
        server: Server,
        env_builder: Callable,
        network_builder: Callable,
        logger_builder: Callable,
        n_episodes: int = 10000,
        batch_size: int = 128,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        annealing_step: int = 100000,
        data_push_freq: int = 5,
        download_weights_freq: int = 10,
    ):
        # This method cannot be executed in other process
        # Note that server cannot be inherited after starting
        self.server = server
        self.env: TSPMDP = None
        self.env_builder = env_builder
        self.encoder: tf.keras.models.Model = None
        self.decoder: tf.keras.models.Model = None
        self.network_builder = network_builder
        self.logger: TFLogger = None
        self.logger_builder = logger_builder
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.annealing_step = annealing_step
        self.data_push_freq = data_push_freq
        self.download_weights_freq = download_weights_freq

    def _initialize(self):
        # Step count
        self.step = 0
        # Build env
        self.env = self.env_builder()
        # Build network instance
        self.encoder, self.decoder = self.network_builder()
        # Build network weights
        self._build()
        # Build logger
        if self.logger_builder:
            self.logger = self.logger_builder()
        else:
            self.logger = None
        # Build local_buffer
        self.local_buffer = defaultdict(list)
        # Set eps
        self.eps = self.eps_start

    def start(self):
        # This method can be executed in subprocess
        self._initialize()
        for episode in range(self.n_episodes):
            self.episode = episode
            metrics = self._episode()
            if self.logger:
                self.logger.log(metrics, episode)
            self._on_episode_end()
    # @tf.function

    def _act(self, decoder_input: List[tf.Tensor]):
        action = self._get_action(decoder_input)
        next_state, reward, done = self.env.step(action)
        return action, reward, next_state, done

    def _get_action(self, state):
        q_values = self.decoder(state)
        # Apply mask
        mask = tf.cast(state[2], tf.float32)
        q_values -= (1-mask) * INFINITY

        greedy_action = tf.argmax(q_values, axis=1, output_type=tf.int32)
        random_action = self._randint(state[2])
        random_flag = tf.cast(tf.random.uniform(
            shape=greedy_action.shape) < self.eps, tf.int32)
        return random_flag * random_action + (1 - random_flag) * greedy_action

    def _episode(self):
        graph, status, mask = self.env.reset()
        graph_embedding = self.encoder(graph)
        done, episode_reward, ones = self._init_episode()
        while tf.math.logical_not(tf.reduce_all(done == ones)):
            # Avoid duplicate encoding
            decoder_input = [graph_embedding, status, mask]
            action, reward, next_state, done = self._act(decoder_input)
            _, next_status, next_mask = next_state
            self._memorize(
                graph=graph,
                status=status,
                mask=mask,
                action=action,
                reward=reward,
                next_status=next_status,
                next_mask=next_mask,
                done=done
            )
            episode_reward += reward
            _, status, mask = next_state
            self._on_step_end()
        metrics = {"episode_reward": tf.reduce_mean(episode_reward)}
        return metrics

    def _memorize(self, **kwargs):

        def append(key, value):
            # Transfer tensor into ndarray
            value = np.array(value)
            self.local_buffer[key].append(value)

        tree.map_structure(append, list(kwargs.keys()), list(kwargs.values()))

    def _build(self):
        graph, status, mask = self.env.reset()
        graph_embedding = self.encoder(graph)
        self.decoder([graph_embedding, status, mask])

    def _init_episode(self):
        shape = (self.batch_size,)
        done = tf.zeros(shape, dtype=tf.int32)
        episode_reward = tf.zeros(shape, dtype=tf.float32)
        ones = tf.ones(shape, dtype=tf.int32)
        return done, episode_reward, ones

    def _on_step_end(self):
        # Count step
        self.step += 1
        # Anneal epsilon
        self._anneal()

    def _on_episode_end(self):
        # Flush the memory
        if (self.episode + 1) % self.data_push_freq == 0:
            self._flush()
        # Download parameters
        if (self.episode + 1) % self.download_weights_freq == 0:
            self._download_weights()

    def _randint(self, mask) -> tf.Tensor:
        """return random action number based on mask

        Args:
            mask (tf.Tensor): B, N

        Returns:
            tf.Tensor: [description]
        """

        uniform = tf.random.uniform(shape=mask.shape)
        actions = tf.math.argmax(
            uniform - INFINITY * tf.cast(1-mask, tf.float32), axis=-1, output_type=tf.int32)
        return actions

    def _anneal(self):
        step = (self.eps_end - self.eps_start) / self.annealing_step
        new_eps = self.eps + step
        if new_eps >= self.eps_end:
            self.eps = new_eps

    def _flush(self):
        def align(value: List[np.ndarray]) -> List[np.ndarray]:
            # Transpose argument value
            stack = np.stack(value)
            shape = (1, 0) + tuple(range(len(stack.shape)))[2:]
            stack = stack.transpose(shape)
            list_of_list = [np.split(val, val.shape[0])
                            for val in stack]
            alignment = []
            for val in list_of_list:
                alignment += (val)
            return alignment

        def dict_transpose(dictionary: dict) -> List[dict]:
            length = len(list(dictionary.values())[0])
            keys = dictionary.keys()
            data = []
            for i in range(length):
                transition = {}
                for key in keys:
                    transition[key] = dictionary[key][i]
                data.append(transition)
            return data

        # Align nested data
        alignment = map_dict(align, self.local_buffer)
        # Split them into list of dictionary
        data = dict_transpose(alignment)
        # Flush them into the server
        self.server.add(data)
        # Clear local buffer
        self.local_buffer = defaultdict(list)

    def _download_weights(self):
        download = self.server.download()
        if download is not None:
            encoder_weights, decoder_weights = download
            self.encoder.set_weights(encoder_weights)
            self.decoder.set_weights(decoder_weights)
