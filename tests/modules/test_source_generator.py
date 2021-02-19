import tensorflow as tf
from tspmdp.modules.source_generator import SourceGenerator
import numpy as np
import pytest


@pytest.mark.parametrize("use_graph_context", [True, False])
def test_source_generator(use_graph_context):
    B, N, D, F1, F2 = 512, 14, 128, 1, 3
    H = tf.ones((B, N, D))
    vector = tf.ones((B, F1))
    indice = tf.constant(np.random.randint(N, size=(B, F2)))
    m = SourceGenerator(use_graph_context=use_graph_context)
    query = m([H, vector, indice])
    assert query.shape == (B, 1, D)
