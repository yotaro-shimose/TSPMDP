from typing import List, Union

import tensorflow as tf
from tspmdp.modules.model import RNDNetwork


class RNDBuilder:
    def __init__(
        self,
        d_model,
        depth,
        n_heads,
        d_key,
        d_hidden,
        n_omega,
        transformer,
        final_ln,
        use_graph_context,
    ):
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_hidden = d_hidden
        self.n_omega = n_omega
        self.transformer = transformer
        self.final_ln = final_ln
        self.use_graph_context = use_graph_context

    def __call__(self):
        builder = RNDNetworkBuilder(
            d_model=self.d_model,
            depth=self.depth,
            n_heads=self.n_heads,
            d_key=self.d_key,
            d_hidden=self.d_hidden,
            n_omega=self.n_omega,
            transformer=self.transformer,
            final_ln=self.final_ln,
            use_graph_context=self.use_graph_context
        )

        rnd = RandomNetworkDistillation(builder)
        return rnd


class RNDNetworkBuilder:
    def __init__(
        self,
        d_model,
        depth,
        n_heads,
        d_key,
        d_hidden,
        n_omega,
        transformer,
        final_ln,
        use_graph_context,
    ):
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_hidden = d_hidden
        self.n_omega = n_omega
        self.transformer = transformer
        self.final_ln = final_ln
        self.use_graph_context = use_graph_context

    def __call__(self):
        return RNDNetwork(
            d_model=self.d_model,
            depth=self.depth,
            n_heads=self.n_heads,
            d_key=self.d_key,
            d_hidden=self.d_hidden,
            n_omega=self.n_omega,
            transformer=self.transformer,
            final_ln=self.final_ln,
            use_graph_context=self.use_graph_context
        )


class RandomNetworkDistillation(tf.keras.models.Model):
    def __init__(self, network_builder: callable, learning_rate=1e-3):
        super().__init__()
        self.random_network: tf.keras.models.Model = network_builder()
        self.exploration_network: tf.keras.models.Model = network_builder()
        self.random_network.trainable = False
        self.loss_function = tf.keras.losses.MSE
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.step_count = tf.Variable(tf.constant(1, dtype=tf.float32))
        self.mean_squared = tf.Variable(tf.constant(0, dtype=tf.float32))
        self.mean = tf.Variable(tf.constant(0, dtype=tf.float32))

    def call(self, observation: Union[List[tf.Tensor], tf.Tensor]):
        if isinstance(observation, list):
            B = observation[0].shape[0]
        else:
            B = observation.shape[0]
        random_embedding = self.random_network(observation)
        new_count = self.step_count + B
        with tf.GradientTape() as tape:
            learning_embedding = self.exploration_network(observation)
            error = self.loss_function(random_embedding, learning_embedding)
            gradient = tape.gradient(
                tf.reduce_mean(error), self.exploration_network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.exploration_network.trainable_variables))
        self.mean_squared.assign(self.mean_squared / new_count *
                                 self.step_count + tf.reduce_sum(error ** 2) / new_count)
        self.mean.assign(self.mean / new_count *
                         self.step_count + tf.reduce_sum(error) / new_count)
        std = self.mean_squared - tf.square(self.mean)

        # Update step count
        self.step_count.assign_add(B)
        # return (error - self.mean) / std
        return error / std
