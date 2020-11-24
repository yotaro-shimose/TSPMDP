import tensorflow as tf
from tspmdp.modules.transformer_block import TransformerBlock, WouterTransformerBlock


class GraphEncoder(tf.keras.models.Model):
    def __init__(self, d_model, depth, n_heads, d_key, d_hidden):
        super().__init__()
        self.transformer = tf.keras.Sequential([
            TransformerBlock(d_model, n_heads, d_key, d_hidden) for _ in range(depth)
        ])
        self.init_dense = tf.keras.layers.Dense(d_model)
        self.final_ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        inputs = self.init_dense(inputs)
        inputs = self.transformer(inputs)
        return self.final_ln(inputs)


class WouterEncoder(tf.keras.models.Model):
    def __init__(self, d_model, depth, n_heads, d_key, d_hidden):
        super().__init__()
        self.transformer = tf.keras.Sequential([
            WouterTransformerBlock(d_model, n_heads, d_key, d_hidden) for _ in range(depth)
        ])
        self.init_dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        inputs = self.init_dense(inputs)
        inputs = self.transformer(inputs)
        return inputs
