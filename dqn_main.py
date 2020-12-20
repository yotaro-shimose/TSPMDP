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
    data_path_list = [
        "./expert_data/100nodes1000samples20201218142925.dat",
        # "./expert_data/100nodes1000samples20201218141310.dat",
    ]
    args = {
        "n_parallels": 16,
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
        "transformer": "gate",
        "final_ln": True,
        "decoder_mha": "softmax",
        "use_graph_context": True,
        "buffer_size": 100000,
        "eps_start": 0.5,
        "eps_end": 0.1,
        "annealing_step": 100000,
        "data_push_freq": 5,
        "download_weights_freq": 5,
        "n_learner_epochs": 1000000,
        "learner_batch_size": 256,
        "learning_rate": 1e-3,
        "upload_freq": 1,
        "sync_freq": 10,
        "scale_value_function": False,
        "logdir": logdir,
        "evaluation_freq": 10,
        "expert_ratio": 0.,
        "data_path_list": data_path_list,
    }
    dqn = TSPDQN(
        **args
    )
    dqn.start()
