import tensorflow as tf
from tspmdp.modules.transformer_block import (
    TransformerBlock, WouterTransformerBlock, GTrXLBlock, LinearTransformerBlock)
from tspmdp.modules.functions import get_args, cut_off


TRANSFORMER_TYPES = {
    "preln": TransformerBlock,
    "gate": GTrXLBlock,
    "linear": LinearTransformerBlock,
    "postbn": WouterTransformerBlock
}


@tf.keras.utils.register_keras_serializable()
class CustomizableEncoder(tf.keras.models.Model):
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        d_key: int,
        d_hidden: int,
        n_omega: int = 64,
        transformer: str = "preln",
        final_ln: bool = True,
        *args,
        **kwargs
    ):
        # Make sure valid transformer type is specified
        assert transformer in TRANSFORMER_TYPES, \
            f"argument 'transformer' should be one of {str(list(TRANSFORMER_TYPES.keys()))}"

        super().__init__(*args, **kwargs)
        # save init arguments
        self.init_args = get_args(offset=1)
        self.init_dense = tf.keras.layers.Dense(d_model)
        TransformerClass = TRANSFORMER_TYPES[transformer]
        # Create arguments for transformer block
        tf_args = {
            "d_model": d_model,
            "depth": depth,
            "n_heads": n_heads,
            "d_key": d_key,
            "d_hidden": d_hidden,
            "n_omega": n_omega
        }
        # Cut off unnecessary arguments
        tf_args = cut_off(TransformerClass.__init__, tf_args)
        self.transformer = tf.keras.Sequential(
            [TransformerClass(**tf_args) for _ in range(depth)])
        self.final_ln = tf.keras.layers.LayerNormalization() if final_ln else None

    def call(self, inputs):
        inputs = self.init_dense(inputs)
        inputs = self.transformer(inputs)
        return self.final_ln(inputs)

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base


class LinearGraphEncoder(tf.keras.models.Model):
    def __init__(self, d_model, depth, n_heads, d_key, d_hidden, n_omega=64):
        super().__init__()
        self.transformer = tf.keras.Sequential([
            LinearTransformerBlock(d_model=d_model,
                                   d_key=d_key,
                                   n_heads=n_heads,
                                   d_hidden=d_hidden,
                                   n_omega=n_omega) for _ in range(depth)
        ])
        self.init_dense = tf.keras.layers.Dense(d_model)
        self.final_ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        inputs = self.init_dense(inputs)
        inputs = self.transformer(inputs)
        return self.final_ln(inputs)


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


class GTrXLEncoder(tf.keras.models.Model):
    def __init__(self, d_model, depth, n_heads, d_key, d_hidden):
        super().__init__()
        self.transformer = tf.keras.Sequential([
            GTrXLBlock(d_model, n_heads, d_key, d_hidden) for _ in range(depth)
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
