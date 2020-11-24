import tensorflow as tf
from typing import List

from tspmdp.modules.graph_encoder import GraphEncoder, WouterEncoder, GTrXLEncoder
from tspmdp.modules.decoder import PolicyDecoder, WouterDecoder


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
