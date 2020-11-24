import tensorflow as tf
from tspmdp.modules.source_generator import SourceGenerator
import numpy as np


def test_source_generator():
    B, N, D, F1, F2 = 512, 14, 128, 1, 3
    H = tf.ones((B, N, D))
    vector = tf.ones((B, F1))
    indice = tf.constant(np.random.randint(N, size=(B, F2)))
    m = SourceGenerator()
    query = m([H, vector, indice])
    assert query.shape == (B, 1, D)
