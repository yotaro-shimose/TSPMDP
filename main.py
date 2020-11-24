from tspmdp.modules.model import GraphAttentionNetwork, WouterAttentionNetwork
from tspmdp.reinforce.reinforce import Reinforce
from tspmdp.logger import TFLogger
import datetime
import pathlib

if __name__ == '__main__':
    d_model = 128
    d_key = 16
    n_heads = 8
    depth = 6
    th_range = 10
    d_hidden = 128

    def network_builder():
        return GraphAttentionNetwork(
            d_model=d_model,
            depth=d_key,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            th_range=th_range
        )
    # def network_builder():
    #     return WouterAttentionNetwork(
    #         d_model=d_model,
    #         depth=d_key,
    #         n_heads=n_heads,
    #         d_key=d_key,
    #         d_hidden=d_hidden,
    #         th_range=th_range
    #     )
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    log_path = str(pathlib.Path("./logs/") / "REINFORCE" / date)
    logger = TFLogger(log_path)

    reinforce = Reinforce(network_builder=network_builder,
                          logger=logger, n_parallels=128, n_nodes=20)
    reinforce.start()
