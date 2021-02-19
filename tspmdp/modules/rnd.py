from typing import List, Union

import tensorflow as tf
from tspmdp.modules.model import RNDNetwork
from tspmdp.modules.functions import get_args


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


@tf.keras.utils.register_keras_serializable()
class RandomNetworkDistillation(tf.keras.models.Model):
    def __init__(self, network_builder: callable, learning_rate: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Random (target) network
        self.random_network: tf.keras.models.Model = network_builder()
        # Exploration network to predict output of random network
        self.exploration_network: tf.keras.models.Model = network_builder()
        # Freeze target network
        self.random_network.trainable = False
        # In case you want to change loss function
        self.loss_function = tf.keras.losses.MSE
        # Optimizer to train exploration network
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, epsilon=1e-4, clipnorm=40)
        # N
        self.step_count = tf.Variable(tf.constant(0, dtype=tf.float32))
        # E[x^2]
        self.mean_squared = tf.Variable(tf.constant(0, dtype=tf.float32))
        # E[x]
        self.mean = tf.Variable(tf.constant(0, dtype=tf.float32))
        # save init arguments
        self.init_args = get_args(offset=1)

    def call(self, observation: Union[List[tf.Tensor], tf.Tensor]):
        # Get batch size (accepts list of tensors in this example)
        if isinstance(observation, list):
            B = observation[0].shape[0]
        else:
            B = observation.shape[0]

        # Compute Target
        random_embedding = self.random_network(observation)
        # New step count will be sum of step count + B
        new_count = self.step_count + B
        with tf.GradientTape() as tape:
            learning_embedding = self.exploration_network(observation)
            error = self.loss_function(random_embedding, learning_embedding)
            gradient = tape.gradient(
                tf.reduce_mean(error), self.exploration_network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.exploration_network.trainable_variables))
        # Update E[x^2]
        self.mean_squared.assign(self.mean_squared / new_count *
                                 self.step_count + tf.reduce_sum(error ** 2) / new_count)
        # Update E[x]
        self.mean.assign(self.mean / new_count *
                         self.step_count + tf.reduce_sum(error) / new_count)
        # Compute V[x]
        var = self.mean_squared - tf.square(self.mean)

        # Update step count
        self.step_count.assign_add(B)
        return (error - self.mean) / tf.sqrt(var)

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base

    @classmethod
    def from_config(cls, config: dict, custom_objects: dict):
        layer = tf.keras.layers.deserialize(config, custom_objects)
        return layer
