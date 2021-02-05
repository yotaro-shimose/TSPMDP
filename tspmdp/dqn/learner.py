import time
from typing import Callable, List, Union

import tensorflow as tf
# import tensorflow_addons as tfa
from tspmdp.dqn.server import ReplayBuffer, Server
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
        soft_sync_ratio: float = None,
        scale_value_function: bool = True,
        expert_ratio: float = 0.,
        replay_buffer_builder: Callable = None,
        data_generator: Callable = None,
        rnd_builder: Callable = None,
        beta: List[float] = None,
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
        self.extrinsic_td_optimizer: tf.keras.optimizers.Optimizer = None
        self.extrinsic_tc_optimizer: tf.keras.optimizers.Optimizer = None
        self.intrinsic_td_optimizer: tf.keras.optimizers.Optimizer = None
        self.intrinsic_tc_optimizer: tf.keras.optimizers.Optimizer = None
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.gamma: Union[List[float], float] = gamma
        self.sync_freq = sync_freq
        self.upload_freq = upload_freq
        self.scale_value_function = scale_value_function
        self.expert_ratio = expert_ratio
        self.data_generator = data_generator
        self.replay_buffer_builder: ReplayBuffer = replay_buffer_builder
        self.use_rnd = rnd_builder is not None
        self.rnd_builder = rnd_builder
        self.rnd: tf.keras.Model = None
        self.beta: Union[List[float], float] = beta
        self.soft_sync_ratio = soft_sync_ratio

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
        self.extrinsic_td_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, epsilon=1e-4, clipnorm=40)
        # self.extrinsic_td_optimizer = tfa.optimizers.RectifiedAdam(
        #     self.learning_rate)
        # self.extrinsic_tc_optimizer = tf.keras.optimizers.Adam(
        #     self.learning_rate)
        # Build logger
        if self.logger_builder:
            self.logger = self.logger_builder()
        # Loss function
        self._loss_function = tf.keras.losses.MSE
        # Expert Buffer
        if self.expert_ratio > 0:
            assert 0 < self.expert_ratio < 1
            assert self.data_generator is not None
            self.expert_buffer = self.replay_buffer_builder()
            data = self.data_generator()
            self.expert_buffer.add(data)
        # Build RND
        if self.use_rnd:
            self._prepare_rnd()

    def _prepare_rnd(self):
        self.rnd = self.rnd_builder()
        self.rnd_encoder, self.rnd_decoder = self.network_builder()
        self.rnd_encoder_target, self.rnd_decoder_target = self.network_builder()
        # M
        assert isinstance(self.beta, list)
        assert isinstance(self.gamma, list)
        self.beta = tf.expand_dims(tf.constant(self.beta), axis=0)
        self.gamma = tf.expand_dims(tf.constant(self.gamma), axis=0)
        self.intrinsic_td_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, epsilon=1e-4, clipnorm=40)
        # self.intrinsic_tc_optimizer = tf.keras.optimizers.Adam(
        #     self.learning_rate)

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

            metrics = self._train_on_batch(**args)
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
        mode=None,  # B, M one-hot vector
    ):
        # n_nodes
        N = mask.shape[-1]
        action = tf.squeeze(action)
        reward = tf.squeeze(reward)
        done = tf.squeeze(done)

        # Compute next action based on Q value
        # B, N
        if self.use_rnd:
            mode = tf.cast(mode, tf.float32)
            next_inputs = graph, next_status, next_mask, mode
            inputs = graph, status, mask, mode

            # B
            beta = tf.reduce_sum(
                self.beta * mode, axis=-1)
            # B
            gamma = tf.reduce_sum(
                self.gamma * mode, axis=-1)
        else:
            next_inputs = graph, next_status, next_mask
            inputs = graph, status, mask
            gamma = self.gamma
            beta = self.beta

        # Q(s, a, j: theta_e)
        # B, N
        next_Q_e_list = self._inference(
            *next_inputs, reward="extrinsic", target=False)
        next_Q_list = tf.identity(next_Q_e_list)
        if self.use_rnd:
            # B, N
            next_Q_i_list = self._inference(
                *next_inputs, reward="intrinsic", target=False)
            next_Q_list = next_Q_list + \
                tf.expand_dims(beta, -1) * next_Q_i_list
        else:
            next_Q_i_list = None

        # B, N
        one_hot_next_action = tf.one_hot(
            tf.math.argmax(next_Q_list, axis=-1), depth=N)

        # Compute extrinsic target
        # B
        next_Q_e = tf.reduce_sum(self._inference(
            *next_inputs, reward="extrinsic", target=True) * one_hot_next_action, axis=-1)
        if self.scale_value_function:
            next_Q_e = rescale(next_Q_e)
        target = reward + gamma ** self.n_step * next_Q_e * \
            (1. - tf.cast(done, tf.float32))
        if self.scale_value_function:
            target = scale(target)

        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables

        # Optimize extrinsic network
        with tf.GradientTape() as tape:
            # Compute TDError
            # B, N
            Q_list = self._inference(*inputs, reward="extrinsic", target=False)
            # B, N
            one_hot_action = tf.one_hot(action, depth=N)
            current_Q = tf.reduce_sum(Q_list * one_hot_action, axis=-1)
            td_loss = tf.reduce_mean(self._loss_function(current_Q, target))

            gradient = tape.gradient(
                td_loss, trainable_variables)
            self.extrinsic_td_optimizer.apply_gradients(
                zip(gradient, trainable_variables))

        # Compute extrinsic TCLoss
        # with tf.GradientTape() as tape:
        #     # B, N
        #     updated_next_Q_list = self._inference(
        #         *next_inputs, reward="extrinsic", target=False)
        #     # B
        #     updated_next_Q = tf.reduce_sum(
        #         updated_next_Q_list * one_hot_next_action, axis=-1)
        #     next_Q = tf.reduce_sum(
        #         next_Q_e_list * one_hot_next_action, axis=-1)
        #     tc_loss = tf.reduce_mean(
        #         self._loss_function(updated_next_Q, next_Q))
        #     gradient = tape.gradient(
        #         tc_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        #     self.extrinsic_tc_optimizer.apply_gradients(zip(
        #         gradient, self.encoder.trainable_variables + self.decoder.trainable_variables))

        metrics = {
            "extrinsic td loss": td_loss,
            # "extrinsic tc loss": tc_loss
            "reward_std": tf.math.reduce_std(reward)
        }

        if self.use_rnd:

            # Compute intrinsic target
            # B
            next_Q_i = tf.reduce_sum(self._inference(
                *next_inputs, reward="intrinsic", target=True) * one_hot_next_action, axis=-1)
            if self.scale_value_function:
                next_Q_i = rescale(next_Q_i)
            # B
            intrinsic_reward = self.rnd([graph, status, mask])
            target = intrinsic_reward + gamma ** self.n_step * next_Q_i * \
                (1. - tf.cast(done, tf.float32))
            if self.scale_value_function:
                target = scale(target)

            trainable_variables = self.rnd_encoder.trainable_variables + \
                self.rnd_decoder.trainable_variables

            # Optimize intrinsic network
            with tf.GradientTape() as tape:
                # Compute TDError
                # B, N
                Q_list = self._inference(
                    *inputs, reward="intrinsic", target=False)
                # B, N
                one_hot_action = tf.one_hot(action, depth=N)
                current_Q = tf.reduce_sum(Q_list * one_hot_action, axis=-1)
                td_loss = tf.reduce_mean(
                    self._loss_function(current_Q, target))

                gradient = tape.gradient(td_loss, trainable_variables)
                self.intrinsic_td_optimizer.apply_gradients(
                    zip(gradient, trainable_variables))

            # # Compute intrinsic TCLoss
            # with tf.GradientTape() as tape:
            #     # B, N
            #     updated_next_Q_list = self._inference(
            #         *next_inputs, reward="intrinsic", target=False)
            #     # B
            #     updated_next_Q = tf.reduce_sum(
            #         updated_next_Q_list * one_hot_next_action, axis=-1)
            #     next_Q = tf.reduce_sum(
            #         next_Q_i_list * one_hot_next_action, axis=-1)
            #     tc_loss = tf.reduce_mean(
            #         self._loss_function(updated_next_Q, next_Q))
            #     gradient = tape.gradient(
            #         tc_loss, trainable_variables)
            #     self.intrinsic_tc_optimizer.apply_gradients(zip(
            #         gradient, trainable_variables))
            intrinsic_metrics = {
                "intrinsic td loss": td_loss,
                # "intrinsic tc loss": tc_loss
            }
            metrics.update(intrinsic_metrics)
        # Return Loss
        return metrics

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
        if self.use_rnd:
            self.rnd_encoder_target.set_weights(self.rnd_encoder.get_weights())
            self.rnd_decoder_target.set_weights(self.rnd_decoder.get_weights())

    def _soft_synchronize(self):
        def weighted_sum(t: List[tf.Tensor], o: List[tf.Tensor]):
            return (1 - self.soft_sync_ratio) * t + self.soft_sync_ratio * o

        def softsync(target: tf.keras.Model, online: tf.keras.Model):
            target_weights = target.get_weights()
            online_weights = online.get_weights()
            new_weights = tf.nest.map_structure(
                weighted_sum, target_weights, online_weights)
            target.set_weights(new_weights)

        softsync(self.encoder_target, self.encoder)
        softsync(self.decoder_target, self.decoder)
        if self.use_rnd:
            softsync(self.rnd_encoder_target, self.rnd_encoder)
            softsync(self.rnd_decoder_target, self.rnd_decoder)

    def _upload(self):
        if self.use_rnd:
            weights = (self.encoder.get_weights(), self.decoder.get_weights(),
                       self.rnd_encoder.get_weights(), self.rnd_decoder.get_weights())
        else:
            weights = self.encoder.get_weights(), self.decoder.get_weights()

        self.server.upload(weights)

    def _on_train_end(self):
        # Step count
        self.step += 1
        # Synchronize weights
        if self.soft_sync_ratio > 0.:
            self._soft_synchronize()
        else:
            if self.step % self.sync_freq == 0:
                self._synchronize()
        # Upload weights
        if self.step % self.upload_freq == 0:
            self._upload()

    def _inference(self, graph, status, mask, mode=None, reward="extrinsic", target=False):
        if reward == "extrinsic":
            if target:
                encoder = self.encoder_target
                decoder = self.decoder_target
            else:
                encoder = self.encoder
                decoder = self.decoder
        elif reward == "intrinsic":
            if target:
                encoder = self.rnd_encoder_target
                decoder = self.rnd_decoder_target
            else:
                encoder = self.rnd_encoder
                decoder = self.rnd_decoder
        else:
            raise ValueError("reward must be either intrinsic or extrinsic")
        graph_embedding = encoder(graph)
        if mode is not None:
            inputs = [graph_embedding, status, mask, mode]
        else:
            inputs = [graph_embedding, status, mask]
        Q = decoder(inputs)
        return Q
