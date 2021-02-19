import tensorflow as tf
from tspmdp.modules.graph_encoder import CustomizableEncoder
from tspmdp.modules.decoder import CustomizableQDecoder
import pathlib
import shutil


def test_save_encoder_decoder():
    depth: int = 2
    n_heads: int = 9
    d_key: int = 9
    n_omega: int = 65
    mha: str = "softmax"
    transformer: str = "preln"
    use_graph_context: bool = False
    accept_mode: bool = True
    final_ln: bool = True
    d_model: int = 129  # for transformer block
    d_hidden: int = 129  # for transformer block
    output_scale: float = -1.  # for tanh activation

    encoder = CustomizableEncoder(
        n_heads=n_heads,
        d_model=d_model,
        depth=depth,
        d_key=d_key,
        d_hidden=d_hidden,
        n_omega=n_omega,
        transformer=transformer,
        final_ln=final_ln,
    )

    decoder = CustomizableQDecoder(
        n_heads=n_heads,
        d_key=d_key,
        n_omega=n_omega,
        mha=mha,
        use_graph_context=use_graph_context,
        accept_mode=accept_mode, d_model=d_model,
        d_hidden=d_hidden,
        output_scale=output_scale
    )

    B = 128
    N = 100
    F1 = 3
    M = 6
    graph = tf.random.normal(shape=(B, N, 2))
    H = encoder(graph)
    status = tf.zeros(shape=(B, F1))
    mask = tf.ones(shape=(B, N))
    mode = tf.one_hot(tf.zeros(shape=(B,), dtype=tf.int32),
                      depth=M, dtype=tf.int32)
    decoder_input = [H, status, mask, mode]
    q_values = decoder(decoder_input)
    assert q_values.shape == (B, N)

    path = pathlib.Path("./dqn/saved_models/tests")
    encoder_path = path / "encoder"
    decoder_path = path / "decoder"
    encoder.save(encoder_path)
    decoder.save(decoder_path)
    loaded_encoder = tf.keras.models.load_model(encoder_path)
    loaded_decoder = tf.keras.models.load_model(decoder_path)
    assert len(loaded_encoder.get_weights()) > 0
    assert len(loaded_decoder.get_weights()) > 0
    loaded_H = loaded_encoder(graph)
    loaded_q_values = loaded_decoder(decoder_input)
    assert H.shape == loaded_H.shape
    assert q_values.shape == loaded_q_values.shape
    shutil.rmtree(path)
