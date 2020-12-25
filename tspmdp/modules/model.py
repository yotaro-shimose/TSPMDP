from typing import List

import tensorflow as tf
from tspmdp.modules.decoder import PolicyDecoder, WouterDecoder
from tspmdp.modules.functions import cut_off, get_args
from tspmdp.modules.graph_encoder import (CustomizableEncoder, GraphEncoder,
                                          GTrXLEncoder, LinearGraphEncoder,
                                          WouterEncoder)
from tspmdp.modules.source_generator import SourceGenerator
from tspmdp.modules.transformer_block import (GTrXLBlock,
                                              LinearTransformerBlock,
                                              TransformerBlock,
                                              WouterTransformerBlock)

TRANSFORMER_TYPES = {
    "preln": TransformerBlock,
    "gate": GTrXLBlock,
    "linear": LinearTransformerBlock,
    "postbn": WouterTransformerBlock
}


@tf.keras.utils.register_keras_serializable()
class RNDNetwork(tf.keras.models.Model):
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
        use_graph_context: bool = True,
        *args,
        **kwargs
    ):
        # Make sure valid transformer type is specified
        assert transformer in TRANSFORMER_TYPES, \
            f"argument 'mha' should be one of {str(list(TRANSFORMER_TYPES.keys()))}"
        super().__init__(*args, **kwargs)
        self.init_args = get_args(offset=1)
        self.graph_encoder = CustomizableEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            n_omega=n_omega,
            transformer=transformer,
            final_ln=final_ln
        )
        self.source_generator = SourceGenerator(
            use_graph_context=use_graph_context, accept_mode=False)
        TransformerClass = TRANSFORMER_TYPES[transformer]
        transformer_args = {
            "n_heads": n_heads,
            "num_heads": n_heads,
            "key_dim": d_key,
            "d_key": d_key,
            "n_omega": n_omega,
            "d_model": d_model,
            "d_hidden": d_hidden,
        }
        transformer_args = cut_off(TransformerClass.__init__, transformer_args)
        self.mha = TransformerClass(**transformer_args)

    def call(self, inputs: List[tf.Tensor]):
        graph, indice, mask = inputs
        H = self.graph_encoder(graph)
        query = self.source_generator([H, indice])

        # B, 1, N
        mask = tf.expand_dims(mask, axis=1)
        # B, 1, D
        embedding = self.mha(query, H, H, mask)
        # B, D
        embedding = tf.squeeze(embedding, axis=1)
        return embedding

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base


class LinearGraphAttentionNetwork(tf.keras.models.Model):

    def __init__(
        self,
        d_model=128,
        depth=6,
        n_heads=8,
        d_key=64,
        d_hidden=128,
        th_range=10,
        n_omega=64
    ):
        super().__init__()
        self.graph_encoder = LinearGraphEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            n_omega=n_omega)
        self.decoder = PolicyDecoder(d_key, th_range)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [graph(B, N, F0), indice(B, F2), mask(B, N)]
        """
        graph = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        H = self.graph_encoder(graph)
        policy = self.decoder([H, indice, mask])
        return policy


class GraphAttentionNetwork(tf.keras.models.Model):
    def __init__(self, d_model=128, depth=6, n_heads=8, d_key=64, d_hidden=128, th_range=10):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            d_model, depth, n_heads, d_key, d_hidden)
        self.decoder = PolicyDecoder(d_key, th_range)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [graph(B, N, F0), indice(B, F2), mask(B, N)]
        """
        graph = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        H = self.graph_encoder(graph)
        policy = self.decoder([H, indice, mask])
        return policy


class GTrXLAttentionNetwork(tf.keras.models.Model):
    def __init__(self, d_model=128, depth=6, n_heads=8, d_key=64, d_hidden=128, th_range=10):
        super().__init__()
        self.graph_encoder = GTrXLEncoder(
            d_model, depth, n_heads, d_key, d_hidden)
        self.decoder = PolicyDecoder(d_key, th_range)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [graph(B, N, F0), indice(B, F2), mask(B, N)]
        """
        graph = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        H = self.graph_encoder(graph)
        policy = self.decoder([H, indice, mask])
        return policy


class WouterAttentionNetwork(tf.keras.models.Model):
    def __init__(self, d_model=128, depth=6, n_heads=8, d_key=64, d_hidden=128, th_range=10):
        super().__init__()
        self.graph_encoder = WouterEncoder(
            d_model, depth, n_heads, d_key, d_hidden)
        self.decoder = WouterDecoder(n_heads, d_key, th_range)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [graph(B, N, F0), indice(B, F2), mask(B, N)]
        """
        graph = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        H = self.graph_encoder(graph)
        policy = self.decoder([H, indice, mask])
        return policy
