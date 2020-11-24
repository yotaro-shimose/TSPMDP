import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_key, d_hidden):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(n_heads, d_key)
        self.bn_1 = tf.keras.layers.LayerNormalization()
        self.bn_2 = tf.keras.layers.LayerNormalization()
        self.dense_1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model, activation='relu')

    def call(self, inputs: tf.Tensor):
        """single transformer layer consisting of
            1. Layer Normalization
            2. Multi Head Self Attention
            3. Residual Connection

        Args:
            inputs (tf.Tensor): Graph Embedding (B, N, D)
        """
        residual = tf.identity(inputs)
        inputs = self.bn_1(inputs)
        inputs = self.mha(inputs, inputs, inputs)
        inputs = residual + inputs
        residual = tf.identity(inputs)
        inputs = self.bn_2(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dense_2(inputs)
        return residual + inputs


class WouterTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_key, d_hidden):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(n_heads, d_key)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.dense_1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs: tf.Tensor):
        """single transformer layer consisting of
            1. Batch Normalization
            2. Multi Head Self Attention
            3. Residual Connection

        Args:
            inputs (tf.Tensor): Graph Embedding (B, N, D)
        """
        residual = tf.identity(inputs)
        inputs = self.mha(inputs, inputs, inputs)
        inputs = residual + inputs
        inputs = self.bn_1(inputs)
        residual = tf.identity(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dense_2(inputs)
        inputs = residual + inputs
        inputs = self.bn_2(inputs)
        return inputs
