import pathlib
import time
from collections import defaultdict, deque
from typing import Callable, List
from copy import deepcopy

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
        eps_end: float = 0.001,
        annealing_step: int = 100000,
        data_push_freq: int = 5,
        download_weights_freq: int = 10,
        evaluation_freq: int = 100,
        save_path: str = None,
        load_path: str = None,
        beta: List[float] = None,
        gamma: List[float] = None,
        ucb_window_size: int = None,
        ucb_eps: float = None,
        ucb_beta: float = None,
    ):
        # This method cannot be executed in other process
        # Note that server cannot be inherited after starting
        self.server = server
        self.env: TSPMDP = None
        self.env_builder = env_builder
        self.encoder: tf.keras.Model = None
        self.decoder: tf.keras.Model = None
        self.rnd_encoder: tf.keras.Model = None
        self.rnd_decoder: tf.keras.Model = None
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
        self.evaluation_freq = evaluation_freq
        self.save_path = save_path
        self.load_path = load_path
        self.best_reward = -INFINITY
        self.beta = beta
        self.gamma = gamma
        self.use_rnd = beta is not None
        self.ucb_beta = ucb_beta
        self.ucb_eps = ucb_eps
        self.ucb: SlidingWindowUCB = None
        self.ucb_window_size = ucb_window_size
        if self.use_rnd:
            assert isinstance(beta, list)
            assert isinstance(gamma, list)
            assert isinstance(ucb_window_size, int)
            assert isinstance(ucb_eps, float)
            assert isinstance(ucb_beta, float)

    def _initialize(self):
        # Episode count
        self.episode = 0
        # Step count
        self.step = 0
        # Build env
        self.env = self.env_builder()
        # Build network instance
        self.encoder, self.decoder = self.network_builder()
        # Build intrinsic network instance
        if self.use_rnd:
            assert isinstance(self.beta, list)
            assert isinstance(self.gamma, list)
            self.beta = tf.constant(self.beta, dtype=tf.float32)
            self.gamma = tf.constant(self.gamma, dtype=tf.float32)
            self.rnd_encoder, self.rnd_decoder = self.network_builder()
            # Build ucb
            self.ucb = SlidingWindowUCB(
                self.beta.shape[0], self.ucb_window_size, self.ucb_beta, self.ucb_eps)
            self.ucb.reset()

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
        # Load network
        if self.load_path is not None:
            self.load(self.load_path)

    def start(self):
        # This method can be executed in subprocess
        self._initialize()
        for episode in range(self.n_episodes):
            metrics = self._episode(training=True)
            if self.logger:
                self.logger.log(metrics, episode)
            self._on_episode_end()

    def _act(
        self,
        decoder_input: List[tf.Tensor],
        training: bool = True,
        rnd_decoder_input: List[tf.Tensor] = None,
    ):
        action = self._get_action(
            decoder_input, training, rnd_decoder_input)
        next_state, reward, done = self.env.step(action)
        return action, reward, next_state, done

    def _get_action(
        self,
        state: List[tf.Tensor],
        training: bool = True,
        rnd_state: List[tf.Tensor] = None,
    ):
        # Compute extrinsic Q values
        Q_list = self.decoder(state)
        # Compute intrinsic Q values when using rnd
        if self.use_rnd:
            Q_i_list = self.rnd_decoder(rnd_state)
            # beta should be zero during evaluation phase
            beta = self.beta * self.ucb.mode if training else 0
        else:
            Q_i_list = tf.zeros(Q_list.shape, dtype=tf.float32)
            beta = 0
        # Log metrics
        if self.logger:
            metrics = {
                "max_Q": tf.reduce_mean(tf.reduce_max(Q_list, axis=-1)),
            }
            if self.use_rnd:
                metrics.update({
                    "max_Q_e": tf.reduce_mean(tf.reduce_max(Q_i_list, axis=-1))
                })
            self.logger.log(metrics, self.step)
        # Compute sum of extrinsic and intrinsic Q values
        Q_list += beta * Q_i_list

        # Compute action using epsilon greedy
        greedy_action = tf.argmax(Q_list, axis=1, output_type=tf.int32)
        random_action = self._randint(state[2])
        random_flag = tf.cast(tf.random.uniform(
            shape=greedy_action.shape) < self.eps, tf.int32)
        if training:
            return random_flag * random_action + (1 - random_flag) * greedy_action
        else:
            return greedy_action

    def _episode(self, training: bool = True):
        graph, status, mask = self.env.reset()
        if self.use_rnd:
            # encode graph using Q(s, a, j: theta_i)
            rnd_graph_embedding = self.rnd_encoder(graph)
        else:
            rnd_graph_embedding = None
        graph_embedding = self.encoder(graph)

        done, episode_reward, ones = self._init_episode()
        while tf.math.logical_not(tf.reduce_all(done == ones)):
            # Avoid duplicate encoding
            decoder_input = [graph_embedding, status, mask]
            inputs = {
                "decoder_input": decoder_input,
                "training": training
            }
            # feed mode vector as decoder input when using rnd
            if self.use_rnd:
                decoder_input.append(self.ucb.mode)
                rnd_decoder_input = [
                    rnd_graph_embedding, status, mask, self.ucb.mode]
                inputs.update(
                    {"rnd_decoder_input": rnd_decoder_input})
            action, reward, next_state, done = self._act(**inputs)
            _, next_status, next_mask = next_state
            if training:
                inputs = {
                    "graph": graph,
                    "status": status,
                    "mask": mask,
                    "action": action,
                    "reward": reward,
                    "next_status": next_status,
                    "next_mask": next_mask,
                    "done": done
                }
                if self.use_rnd:
                    inputs.update({
                        "mode": self.ucb.mode
                    })
                self._memorize(
                    **inputs
                )
            episode_reward += reward
            _, status, mask = next_state
            if training:
                self._on_step_end()
        if training:
            # execute ucb step
            if self.ucb is not None:
                self.ucb.step(episode_reward)
            metrics = {
                "training: episode_reward": tf.reduce_mean(episode_reward),
                "eps": self.eps
            }
        else:
            metrics = {
                "evaluation: episode_reward": tf.reduce_mean(episode_reward),
            }
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
        if self.use_rnd:
            graph, status, mask = self.env.reset()
            graph_embedding = self.rnd_encoder(graph)
            self.rnd_decoder([graph_embedding, status, mask])

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
        # Count step
        self.episode += 1
        # Flush the memory
        if (self.episode + 1) % self.data_push_freq == 0:
            self._flush()
        # Download parameters
        if (self.episode + 1) % self.download_weights_freq == 0:
            self._download_weights()
        # Execute evaluation and save the best model
        if (self.episode + 1) % self.evaluation_freq == 0:
            metrics = self._episode(training=False)
            if self.logger:
                self.logger.log(metrics, self.episode)
            episode_reward = metrics["evaluation: episode_reward"]
            if episode_reward > self.best_reward and self.save_path is not None:
                self.save(self.save_path)

        # Wait for a second so that replay buffer won't be overwhelmed by actor's request
        time.sleep(1)

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

    def save(self, path: str):
        encoder_path = pathlib.Path(path) / "encoder"
        decoder_path = pathlib.Path(path) / "decoder"
        self.encoder.save_weights(encoder_path)
        self.decoder.save_weights(decoder_path)

    def load(self, path: str):
        if not self.encoder.built or not self.decoder.built:
            self._build()
        encoder_path = pathlib.Path(path) / "encoder"
        decoder_path = pathlib.Path(path) / "decoder"
        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)


class SlidingWindowUCB:
    def __init__(self, n_modes: tf.TensorShape, window_size: int, beta: float, eps: float):
        self._mode = np.eye(n_modes)[0]
        self.beta = beta
        self.eps = eps
        self.episode_rewards = deque(maxlen=window_size)
        self.modes = deque(maxlen=window_size)

    def reset(self):
        return self._mode

    def step(self, episode_reward: tf.Tensor):
        batch_size = episode_reward.shape[0]
        # B
        episode_reward = np.array(episode_reward)
        # List of (M,)
        episode_rewards = [
            reward * self.mode for reward in np.split(episode_reward, batch_size)]
        # List of (M,)
        modes = [self.mode for _ in range(batch_size)]
        self._append(episode_rewards, modes)
        # Update mode using epsilon greedy algorithm
        n_modes = self.modes.shape[0]
        if np.random.random() < self.eps:
            self._mode = np.eye(n_modes)[np.random.randint(n_modes)]
        else:
            # M
            rewards = np.sum(np.stack(self.episode_rewards), axis=0)
            # M
            modes = np.sum(np.stack(self.modes), axis=0)
            # M
            mean_rewards = rewards / (modes + 1)
            # M
            exploration_term = np.sqrt(np.log(np.sum(modes)) / modes)
            # M
            score = mean_rewards + self.beta * exploration_term
            # M
            self._mode = np.eye(n_modes)[np.argmax(score)]

    def _append(self, rewards: List[np.ndarray], modes: List[np.ndarray]):
        self.episode_rewards.extend(rewards)
        self.modes.extend(modes)

    @property
    def mode(self) -> np.ndarray:
        return deepcopy(self._mode)
