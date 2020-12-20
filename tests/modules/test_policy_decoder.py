from tspmdp.modules.decoder import PolicyDecoder, WouterDecoder, CustomizablePolicyDecoder,\
    CustomizableQDecoder
import tensorflow as tf
import numpy as np
import pytest


def test_policy_decoder():
    decoder = PolicyDecoder(512, 10)
    B, N, D, F = 512, 14, 128, 3
    H = tf.ones((B, N, D))
    indice = tf.constant(np.random.randint(N, size=(B, F)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = decoder([H, indice, mask])
    tf.assert_equal(policy * (1-mask),
                    tf.zeros(policy.shape))


def test_wouter_decoder():
    decoder = WouterDecoder(8, 512, 10)
    B, N, D, F = 512, 14, 128, 3
    H = tf.ones((B, N, D))
    indice = tf.constant(np.random.randint(N, size=(B, F)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = decoder([H, indice, mask])
    tf.assert_equal(policy * (1-mask),
                    tf.zeros(policy.shape))


@pytest.mark.parametrize("use_graph_context", [True, False])
# @pytest.mark.parametrize("mha", ["softmax", "linear", "none"])
@pytest.mark.parametrize("mha", ["softmax", "none"])
def test_customizable__policy_decoder(mha: str, use_graph_context: bool):
    decoder = CustomizablePolicyDecoder(
        8, 512, 10, mha=mha, use_graph_context=use_graph_context)
    B, N, D, F = 512, 14, 128, 3
    H = tf.ones((B, N, D))
    indice = tf.constant(np.random.randint(N, size=(B, F)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = decoder([H, indice, mask])
    tf.assert_equal(policy * (1-mask),
                    tf.zeros(policy.shape))


@pytest.mark.parametrize("use_graph_context", [True, False])
# @pytest.mark.parametrize("mha", ["softmax", "linear", "none"])
@pytest.mark.parametrize("mha", ["softmax", "none"])
def test_customizable_Q_decoder(mha: str, use_graph_context: bool):
    decoder = CustomizableQDecoder(
        8, 512, 10, mha=mha, use_graph_context=use_graph_context)
    B, N, D, F = 512, 14, 128, 3
    H = tf.ones((B, N, D))
    indice = tf.constant(np.random.randint(N, size=(B, F)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = decoder([H, indice, mask])
    assert policy.shape == (B, N)
