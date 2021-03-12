import datetime
import pathlib
import tensorflow as tf


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
            tf.config.experimental.set_memory_growth(
                physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")


if __name__ == '__main__':
    allocate_gpu_memory()

    from tspmdp.logger import TFLogger
    from tspmdp.modules.graph_encoder import \
        LinearGraphEncoder, GraphEncoder, GTrXLEncoder, WouterEncoder
    from tspmdp.modules.decoder import PolicyDecoder, WouterDecoder
    from tspmdp.reinforce.reinforce import Reinforce

    d_model = 128
    d_key = 16
    n_heads = 8
    depth = 6
    th_range = 10
    d_hidden = 128
    n_nodes = 20
    n_parallels = 64
    n_omega = 128
    encoder = "wouter"
    decoder = "wouter"

    def preln():
        return GraphEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden
        )

    def linear():
        return LinearGraphEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            n_omega=n_omega
        )

    def gate():
        return GTrXLEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden
        )

    def wouter():
        return WouterEncoder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden
        )

    encoder_builders = {
        "preln": preln,
        "linear": linear,
        "gate": gate,
        "wouter": wouter
    }
    encoder_builder = encoder_builders[encoder]

    def projection():
        return PolicyDecoder(d_key=d_key, th_range=th_range)

    def wouter_dec():
        return WouterDecoder(n_heads=n_heads, d_key=d_key, th_range=th_range)

    decoder_builders = {
        "projection": projection,
        "wouter": wouter_dec
    }
    decoder_builder = decoder_builders[decoder]

    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    log_path = str(pathlib.Path("./logs/") / "REINFORCE" /
                   ("nodes" + str(n_nodes)) / date)
    logger = TFLogger(log_path)

    reinforce = Reinforce(encoder_builder=encoder_builder, decoder_builder=decoder_builder,
                          logger=logger, n_parallels=n_parallels, n_nodes=n_nodes)
    reinforce.start()
