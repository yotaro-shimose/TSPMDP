import tensorflow as tf
from tspmdp.modules.graph_encoder import CustomizableEncoder, GraphEncoder, WouterEncoder
import pytest
import shutil


def test_graph_encoder():
    B, N, F = 512, 14, 2
    n_heads = 8
    d_key = 64
    d_hidden = 256
    depth = 6
    d_model = 128
    model = GraphEncoder(d_model, depth, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, F))
    assert model(inputs).shape == (B, N, d_model)


def test_woter_graph_encoder():
    B, N, F = 512, 14, 2
    n_heads = 8
    d_key = 64
    d_hidden = 256
    depth = 6
    d_model = 128
    model = WouterEncoder(d_model, depth, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, F))
    assert model(inputs).shape == (B, N, d_model)


@pytest.mark.parametrize("final_ln", [True, False])
@pytest.mark.parametrize("transformer", ["preln", "gate", "linear", "postbn"])
def test_customizable_encoder(transformer, final_ln):
    B, N, F = 512, 14, 2
    n_heads = 8
    d_key = 64
    d_hidden = 256
    depth = 6
    d_model = 128
    model = CustomizableEncoder(d_model, depth, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, F))
    assert model(inputs).shape == (B, N, d_model)
