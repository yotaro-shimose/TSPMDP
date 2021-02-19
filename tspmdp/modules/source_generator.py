from typing import List

import tensorflow as tf
from tspmdp.modules.functions import get_args


@tf.keras.utils.register_keras_serializable()
class SourceGenerator(tf.keras.layers.Layer):

    def __init__(self, use_graph_context: bool = True, accept_mode: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.use_graph_context = use_graph_context
        self.ln = tf.keras.layers.LayerNormalization()
        self.accept_mode = accept_mode
        # save init arguments
        self.init_args = get_args(offset=1)

    def build(self, input_shape):
        _, _, D = input_shape[0]
        self.final_dense = tf.keras.layers.Dense(D, activation='relu')
        if self.accept_mode:
            self.mode_dense = tf.keras.layers.Dense(D, activation='relu')
        else:
            self.mode_dense = None
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        """transfer inputs into the form of [Source, Target] which is the input form of Decoder

        Args:
            inputs (List[tf.Tensor]): [graph_embedding(B, N, D), indice(B, F)]

        Returns:
            query (tf.Tensor): (B, 1, N) query for source target attention
        """

        # B, N, D
        H = inputs[0]
        # B, F
        indice = inputs[1]

        # B, F, D
        indice_embedding = self._indice_embedding(H, indice)

        if self.use_graph_context:
            # B, 1, D
            graph_embedding = tf.reduce_mean(H, axis=-2, keepdims=True)

            # B, (D * (F + 1))
            embedding = self.flatten(
                tf.concat([indice_embedding, graph_embedding], axis=-2))
        else:
            # B, D
            embedding = self.flatten(indice_embedding)

        if self.accept_mode:
            # B, M
            mode = inputs[2]
            # B, D
            mode_embedding = self.mode_dense(mode)
            # B, *
            embedding = tf.concat([embedding, mode_embedding], axis=-1)

        # B, *
        embedding = self.final_dense(embedding)
        embedding = self.ln(embedding)
        return tf.expand_dims(embedding, -2)

    def _indice_embedding(self, H, indice):
        indice = tf.cast(indice, tf.int32)
        # calculate indice embeddings

        # B, F, N
        indice = tf.one_hot(indice, depth=H.shape[1])
        # B, F, N, 1
        indice = tf.expand_dims(indice, -1)
        # B, F, N, D
        indice = tf.tile(indice, [1, 1, 1, H.shape[-1]])

        # B, 1, N, D
        H = tf.expand_dims(H, 1)
        # B, F, N, D
        H = tf.tile(H, [1, indice.shape[1], 1, 1])

        # B, F, D
        indice_embeddings = tf.reduce_sum(indice * H, -2)
        return indice_embeddings

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base


@tf.keras.utils.register_keras_serializable()
class WouterSourceGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # save init arguments
        self.init_args = get_args(offset=1)

    def build(self, input_shape):
        _, _, D = input_shape[0]
        self.final_dense = tf.keras.layers.Dense(
            D, activation='relu', use_bias=False)
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        """transfer inputs into the form of [Source, Target] which is the input form of Decoder

        Args:
            inputs (List[tf.Tensor]): [graph_embedding(B, N, D), indice(B, F)]

        Returns:
            query (tf.Tensor): (B, 1, N) query for source target attention
        """

        # B, N, D
        H = inputs[0]
        # B, F
        indice = inputs[1]

        # B, F, D
        indice_embedding = self._indice_embedding(H, indice)

        # B, 1, D
        graph_embedding = tf.reduce_mean(H, axis=-2, keepdims=True)

        # B, (D * (F + 1))
        embedding = self.flatten(
            tf.concat([indice_embedding, graph_embedding], axis=-2))
        # B, D
        embedding = self.final_dense(embedding)
        # B, 1, D
        return tf.expand_dims(embedding, -2)

    def _indice_embedding(self, H, indice):
        indice = tf.cast(indice, tf.int32)
        # calculate indice embeddings

        # B, F, N
        indice = tf.one_hot(indice, depth=H.shape[1])
        # B, F, N, 1
        indice = tf.expand_dims(indice, -1)
        # B, F, N, D
        indice = tf.tile(indice, [1, 1, 1, H.shape[-1]])

        # B, 1, N, D
        H = tf.expand_dims(H, 1)
        # B, F, N, D
        H = tf.tile(H, [1, indice.shape[1], 1, 1])

        # B, F, D
        indice_embeddings = tf.reduce_sum(indice * H, -2)
        return indice_embeddings

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base
