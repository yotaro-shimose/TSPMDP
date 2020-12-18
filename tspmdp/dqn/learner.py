from typing import Callable

import tensorflow as tf
import time
from tspmdp.dqn.server import Server
from tspmdp.logger import TFLogger

INFINITY = 1e+9


def scale(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    return tf.sign(x) * (tf.sqrt(abs(x) + 1) - 1) + eps * x


def rescale(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    z = tf.sqrt(1 + 4 * eps * (eps + 1 + abs(x))) / 2 / eps - 1 / 2 / eps
    return tf.sign(x) * (tf.square(z) - 1)


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
        scale_value_function=True
    ):
        self.server = server
        self.network_builder = network_builder
        self.encoder: tf.keras.Model = None
        self.decoder: tf.keras.Model = None
        self.encoder_target: tf.keras.Model = None
        self.decoder_target: tf.keras.Model = None
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

    def start(self):
        self._initialize()
        for epoch in range(self.n_epochs):
            metrics = self._train()
            if self.logger:
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
        if self.logger:
            self.logger = self.logger_builder()
        # Loss function
        self._loss_function = tf.keras.losses.Huber()

    def _train(self):
        batch = self.server.sample(self.batch_size)
        if batch:
            args = self._create_args(batch)
            raw_metrics = self._train_on_batch(**args)
            metrics = self._create_metrics(raw_metrics)
            self._on_train_end()
            return metrics

        else:
            time.sleep(1)

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
        done
    ):
        # n_nodes
        N = mask.shape[-1]
        action = tf.squeeze(action)
        reward = tf.squeeze(reward)
        done = tf.squeeze(done)

        # Compute Q Target
        # B, N
        next_Q_list = self._inference(graph, next_status, next_mask)
        # Apply mask
        next_Q_list -= tf.cast(1 - mask, tf.float32) * INFINITY
        # B, N
        one_hot_next_action = tf.one_hot(
            tf.math.argmax(next_Q_list, axis=-1), depth=N)
        # B
        next_Q = tf.reduce_sum(self._inference_target(
            graph, next_status, next_mask) * one_hot_next_action, axis=-1)
        if self.scale_value_function:
            next_Q = rescale(next_Q)
        target = reward + self.gamma ** self.n_step * next_Q * \
            (1. - tf.cast(done, tf.float32))
        if self.scale_value_function:
            target = scale(target)

        with tf.GradientTape() as tape:
            # Calculate TDError
            # B, N
            Q_list = self._inference(graph, status, mask)
            # B, N
            one_hot_action = tf.one_hot(action, depth=N)
            current_Q = tf.reduce_sum(Q_list * one_hot_action, axis=-1)
            if self.scale_value_function:
                current_Q = scale(current_Q)
            loss = tf.reduce_mean(self._loss_function(current_Q, target))

            gradient = tape.gradient(loss, self._trainable_variables())
            self.optimizer.apply_gradients(
                zip(gradient, self._trainable_variables()))

        # Return Loss
        return loss

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
        return args

    def _synchronize(self):
        self.encoder_target.set_weights(self.encoder.get_weights())
        self.decoder_target.set_weights(self.decoder.get_weights())

    @property
    def built(self):
        return self.encoder.built\
            and self.decoder.built and self.encoder_target.built and self.decoder_target.built

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

    def _inference(self, graph, status, mask):
        graph_embedding = self.encoder(graph)
        Q = self.decoder([graph_embedding, status, mask])
        return Q

    def _inference_target(self, graph, status, mask):
        graph_embedding = self.encoder_target(graph)
        Q = self.decoder_target([graph_embedding, status, mask])
        return Q

    def _create_metrics(self, raw_metrics):
        return {"loss": raw_metrics}
