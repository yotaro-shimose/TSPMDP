from tspmdp.modules.model import GraphAttentionNetwork, WouterAttentionNetwork
import tensorflow as tf
import numpy as np


def test_graph_attention_network():
    model = GraphAttentionNetwork()
    B, N, F0, F1 = 512, 14, 5, 3
    graph = tf.ones((B, N, F0))
    indice = tf.constant(np.random.randint(N, size=(B, F1)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = model([graph, indice, mask])
    tf.assert_equal(policy * (1-mask),
                    tf.zeros(policy.shape))
    assert policy.shape == (B, N)


def test_wouter_attention_network():
    model = WouterAttentionNetwork()
    B, N, F0, F1 = 512, 14, 5, 3
    graph = tf.ones((B, N, F0))
    indice = tf.constant(np.random.randint(N, size=(B, F1)), dtype=tf.int32)
    mask = tf.constant(np.random.randint(2, size=(B, N)), dtype=tf.float32)
    policy = model([graph, indice, mask])
    tf.assert_equal(policy * (1-mask),
                    tf.zeros(policy.shape))
    assert policy.shape == (B, N)
