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
    from tspmdp.modules.model import (GraphAttentionNetwork, GTrXLAttentionNetwork,
                                      WouterAttentionNetwork, LinearGraphAttentionNetwork)
    from tspmdp.reinforce.reinforce import Reinforce

    d_model = 128
    d_key = 16
    n_heads = 8
    depth = 6
    th_range = 30
    d_hidden = 128
    n_nodes = 100
    n_parallels = 32
    n_omega = 128

    # def network_builder():
    #     return LinearGraphAttentionNetwork(
    #         n_omega=n_omega,
    #         d_model=d_model,
    #         depth=depth,
    #         n_heads=n_heads,
    #         d_key=d_key,
    #         d_hidden=d_hidden,
    #         th_range=th_range
    #     )
    def network_builder():
        return GraphAttentionNetwork(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            th_range=th_range
        )
    # def network_builder():
    #     return WouterAttentionNetwork(
    #         d_model=d_model,
    #         depth=depth,
    #         n_heads=n_heads,
    #         d_key=d_key,
    #         d_hidden=d_hidden,
    #         th_range=th_range
    #     )
    # def network_builder():
    #     return GTrXLAttentionNetwork(
    #         d_model=d_model,
    #         depth=depth,
    #         n_heads=n_heads,
    #         d_key=d_key,
    #         d_hidden=d_hidden,
    #         th_range=th_range
    #     )

    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    log_path = str(pathlib.Path("./logs/") / "REINFORCE" /
                   ("nodes" + str(n_nodes)) / date)
    logger = TFLogger(log_path)

    reinforce = Reinforce(network_builder=network_builder,
                          logger=logger, n_parallels=n_parallels, n_nodes=n_nodes)
    reinforce.start()
