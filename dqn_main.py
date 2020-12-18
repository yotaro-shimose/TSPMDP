import datetime
import pathlib
import multiprocessing
from tspmdp.dqn.dqn import TSPDQN

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    n_nodes = 100
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    logdir = str(pathlib.Path("./logs/") / "DQN" /
                 ("nodes" + str(n_nodes)) / date)
    args = {
        "n_parallels": 32,
        "n_nodes": n_nodes,
        "n_episodes": 100000,
        "n_step": 3,
        "gamma": 0.999,
        "d_model": 128,
        "depth": 6,
        "n_heads": 8,
        "d_key": 16,
        "d_hidden": 128,
        "n_omega": 64,
        "transformer": "preln",
        "final_ln": True,
        "decoder_mha": "softmax",
        "use_graph_context": True,
        "buffer_size": 1000000,
        "eps_start": 1,
        "eps_end": 0.01,
        "annealing_step": 1000000,
        "data_push_freq": 5,
        "download_weights_freq": 50,
        "n_learner_epochs": 1000000,
        "learner_batch_size": 64,
        "learning_rate": 1e-3,
        "upload_freq": 100,
        "sync_freq": 50,
        "scale_value_function": False,
        "logdir": logdir
    }
    dqn = TSPDQN(
        **args
    )
    dqn.start()
