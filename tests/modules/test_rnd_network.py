import pytest
import tensorflow as tf

from tspmdp.modules.model import RNDNetwork


@pytest.mark.parametrize("d_model", [64])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("d_key", [8])
@pytest.mark.parametrize("d_hidden", [64])
@pytest.mark.parametrize("n_omega", [64])
@pytest.mark.parametrize("transformer", ["preln"])
@pytest.mark.parametrize("final_ln", [True])
@pytest.mark.parametrize("use_graph_context", [True])
def test_rnd_network(
    d_model,
    depth,
    n_heads,
    d_key,
    d_hidden,
    n_omega,
    transformer,
    final_ln,
    use_graph_context
):

    B, D, N = 128, d_model, 100
    model = RNDNetwork(
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_key=d_key,
        d_hidden=d_hidden,
        n_omega=n_omega,
        transformer=transformer,
        final_ln=final_ln,
        use_graph_context=use_graph_context,
    )
    inputs = [tf.random.uniform((B, N, 2)), tf.ones(
        (B, 2), dtype=tf.int32), tf.ones((B, N), dtype=tf.int32)]
    output = model(inputs)
    assert output.shape == (B, D)
