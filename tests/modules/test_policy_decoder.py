from tspmdp.modules.decoder import PolicyDecoder, WouterDecoder
import tensorflow as tf
import numpy as np


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


if __name__ == '__main__':
    test_wouter_decoder()
