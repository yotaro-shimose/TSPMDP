from typing import Callable

import tensorflow as tf
import time
from tspmdp.dqn.server import Server, ReplayBuffer
from tspmdp.logger import TFLogger

INFINITY = 1e+9


def scale(x: tf.Tensor, eps: float = 1e-2) -> tf.Tensor:
    return tf.sign(x) * (tf.sqrt(abs(x) + 1) - 1) + eps * x


def rescale(x: tf.Tensor, eps: float = 1e-2) -> tf.Tensor:
    z = tf.sqrt(1 + 4 * eps * (eps + 1 + abs(x))) / 2 / eps - 1 / 2 / eps
    return tf.sign(x) * (tf.square(z) - 1)


def concat(x: tf.Tensor, y: tf.Tensor):
    return tf.concat([x, y], axis=0)


class Learner:

    def __init__(
        self,
        server: Server,
        network_builder: Callable,
        logger_builder: Callable,
        n_epochs: int = 1000000,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        n_step=3,
        gamma=0.9999,
        upload_freq: int = 100,
        sync_freq: int = 50,
        scale_value_function: bool = True,
        expert_ratio: float = 0.,
        replay_buffer_builder: Callable = None,
        data_generator: Callable = None,
        rnd_builder: Callable = None,
    ):
        self.server = server
        self.network_builder = network_builder
        self.encoder: tf.keras.Model = None
        self.decoder: tf.keras.Model = None
        self.encoder_target: tf.keras.Model = None
        self.decoder_target: tf.keras.Model = None
        self.rnd_encoder: tf.keras.Model = None
        self.rnd_decoder: tf.keras.Model = None
        self.rnd_encoder_target: tf.keras.Model = None
        self.rnd_decoder_target: tf.keras.Model = None
        self.logger: TFLogger = None
        self.logger_builder = logger_builder
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.gamma = gamma
        self.sync_freq = sync_freq
        self.upload_freq = upload_freq
        self.scale_value_function = scale_value_function
        self.expert_ratio = expert_ratio
        self.data_generator = data_generator
        self.replay_buffer_builder: ReplayBuffer = replay_buffer_builder
        self.rnd_builder = rnd_builder
        self.rnd: tf.keras.Model = None

    def start(self):

        self._initialize()
        for epoch in range(self.n_epochs):
            metrics = self._train()
            if self.logger is not None and metrics is not None:
                self.logger.log(metrics, epoch)

    def _initialize(self):
        # Step count
        self.step = 0
        # Build network instances
        self.encoder, self.decoder = self.network_builder()
        self.encoder_target, self.decoder_target = self.network_builder()
        # Build Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        # Build logger
        if self.logger_builder:
            self.logger = self.logger_builder()
        # Loss function
        self._loss_function = tf.keras.losses.Huber()
        # Expert Buffer
        if self.expert_ratio > 0:
            assert 0 < self.expert_ratio < 1
            assert self.data_generator is not None
            self.expert_buffer = self.replay_buffer_builder()
            data = self.data_generator()
            self.expert_buffer.add(data)
        # Build RND
        if self.rnd_builder:
            self.rnd = self.rnd_builder()
            self.rnd_encoder, self.rnd_decoder = self.network_builder()
            self.rnd_encoder_target, self.rnd_decoder_target = self.network_builder()

    def _train(self):
        expert_batch_size = int(self.batch_size * self.expert_ratio)
        batch_size = self.batch_size - expert_batch_size
        batch = self.server.sample(batch_size)
        if batch:
            args = self._create_args(batch)
            if self.expert_ratio > 0:
                expert_batch = self.expert_buffer.sample(expert_batch_size)
                assert expert_batch is not None
                expert_args = self._create_args(expert_batch)
                args = tf.nest.map_structure(concat, args, expert_args)

            raw_metrics = self._train_on_batch(**args)
            metrics = self._create_metrics(raw_metrics)
            self._on_train_end()
            return metrics
        else:
            time.sleep(1)
            return None

    @tf.function
    def _train_on_batch(
        self,
        graph,
        status,
        mask,
        action,
        reward,
        next_status,
        next_mask,
        done,
        mode=None,
    ):
        # n_nodes
        N = mask.shape[-1]
        action = tf.squeeze(action)
        reward = tf.squeeze(reward)
        done = tf.squeeze(done)

        # Compute Q Target
        # B, N
        if mode:
            next_inputs = graph, next_status, next_mask, mode
            inputs = graph, status, mask, mode
        else:
            next_inputs = graph, next_status, next_mask
            inputs = graph, status, mask
        next_Q_list = self._inference(next_inputs)
        # B, N
        one_hot_next_action = tf.one_hot(
            tf.math.argmax(next_Q_list, axis=-1), depth=N)
        # B
        next_Q = tf.reduce_sum(self._inference_target(
            next_inputs) * one_hot_next_action, axis=-1)
        if self.scale_value_function:
            next_Q = rescale(next_Q)
        target = reward + self.gamma ** self.n_step * next_Q * \
            (1. - tf.cast(done, tf.float32))
        if self.scale_value_function:
            target = scale(target)

        with tf.GradientTape() as tape:
            # Compute TDError
            # B, N
            Q_list = self._inference(inputs)
            # B, N
            one_hot_action = tf.one_hot(action, depth=N)
            current_Q = tf.reduce_sum(Q_list * one_hot_action, axis=-1)
            if self.scale_value_function:
                current_Q = scale(current_Q)
            td_loss = tf.reduce_mean(self._loss_function(current_Q, target))

            gradient = tape.gradient(td_loss, self._trainable_variables())
            self.optimizer.apply_gradients(
                zip(gradient, self._trainable_variables()))

        # Compute TCLoss
        with tf.GradientTape() as tape:
            # B, N
            updated_next_Q_list = self._inference(
                next_inputs)
            # B
            updated_next_Q = tf.reduce_sum(
                updated_next_Q_list * one_hot_next_action, axis=-1)
            tc_loss = tf.reduce_mean(self._loss_function(
                updated_next_Q, tf.reduce_max(next_Q_list, axis=-1)))
            gradient = tape.gradient(tc_loss, self._trainable_variables())
            self.optimizer.apply_gradients(
                zip(gradient, self._trainable_variables()))

        # Return Loss
        return td_loss, tc_loss

    def _create_args(self, batch: dict):
        args = {
            "graph": tf.constant(batch["graph"], dtype=tf.float32),
            "status": tf.constant(batch["status"], dtype=tf.int32),
            "mask": tf.constant(batch["mask"], dtype=tf.int32),
            "action": tf.constant(batch["action"], dtype=tf.int32),
            "reward": tf.constant(batch["reward"], dtype=tf.float32),
            "next_status": tf.constant(batch["status"], dtype=tf.int32),
            "next_mask": tf.constant(batch["next_mask"], dtype=tf.int32),
            "done": tf.constant(batch["done"], dtype=tf.int32)
        }
        if self.rnd:
            args["mode"] = tf.constant(batch["mode"], dtype=tf.int32)
        return args

    def _synchronize(self):
        self.encoder_target.set_weights(self.encoder.get_weights())
        self.decoder_target.set_weights(self.decoder.get_weights())

    def _upload(self):
        weights = self.encoder.get_weights(), self.decoder.get_weights()
        self.server.upload(weights)

    def _on_train_end(self):
        # Step count
        self.step += 1
        # Synchronize weights
        if self.step % self.sync_freq == 0:
            self._synchronize()
        # Upload weights
        if self.step % self.upload_freq == 0:
            self._upload()

    def _trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def _inference(self, graph, status, mask, mode=None):
        graph_embedding = self.encoder(graph)
        if mode:
            inputs = [graph_embedding, status, mask, mode]
        else:
            inputs = [graph_embedding, status, mask]
        Q = self.decoder(inputs)
        return Q

    def _inference_target(self, graph, status, mask, mode=None):
        graph_embedding = self.encoder_target(graph)
        if mode:
            inputs = [graph_embedding, status, mask, mode]
        else:
            inputs = [graph_embedding, status, mask]
        Q = self.decoder_target(inputs)
        return Q

    def _create_metrics(self, raw_metrics):
        td_loss, tc_loss = raw_metrics
        return {
            "td_loss": td_loss,
            "tc_loss": tc_loss
        }
