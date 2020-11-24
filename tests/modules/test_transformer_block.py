import tensorflow as tf

from tspmdp.modules.transformer_block import TransformerBlock, WouterTransformerBlock, GTrXLBlock


def test_transformer_block():
    B, N, D = 512, 14, 128
    n_heads = 8
    d_key = 64
    d_hidden = 256
    layer = TransformerBlock(D, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, D))
    assert layer(inputs).shape == (B, N, D)


def test_wouter_transformer_block():
    B, N, D = 512, 14, 128
    n_heads = 8
    d_key = 64
    d_hidden = 256
    layer = WouterTransformerBlock(D, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, D))
    assert layer(inputs).shape == (B, N, D)


def test_gtrxl_block():
    B, N, D = 512, 14, 128
    n_heads = 8
    d_key = 64
    d_hidden = 256
    layer = GTrXLBlock(D, n_heads, d_key, d_hidden)
    inputs = tf.ones((B, N, D))
    assert layer(inputs).shape == (B, N, D)


if __name__ == '__main__':
    test_gtrxl_block()
