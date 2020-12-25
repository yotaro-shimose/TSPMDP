import datetime
import pathlib
import multiprocessing
from tspmdp.dqn.dqn import TSPDQN


# TODO
# - create RND using transformer
# - make learner to learn separate network for extrinsic and intrinsic reward respectively using
# gamma stored in buffer
# - make actor to use separate network for extrinsic and intrinsic reward
# - make actor to store exploration mode
# - make actor to choose exploration mode using sliding UCB
# - parametrize to switch RND

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
    save_path = None
    args = {
        "n_parallels": 16,
        "n_nodes": n_nodes,
        "n_episodes": 100000,
        "n_step": 1,
        "gamma": 0.99,
        "d_model": 128,
        "depth": 3,
        "n_heads": 8,
        "d_key": 16,
        "d_hidden": 128,
        "n_omega": 64,
        "transformer": "gate",
        "final_ln": True,
        "decoder_mha": "gate",
        "use_graph_context": False,
        "buffer_size": 100000,
        "eps_start": 0.5,
        "eps_end": 0.15,
        "annealing_step": 100000,
        "data_push_freq": 5,
        "download_weights_freq": 5,
        "n_learner_epochs": 1000000,
        "learner_batch_size": 128,
        "learning_rate": 1e-4,
        "upload_freq": 100,
        "sync_freq": 100,
        "scale_value_function": False,
        "logdir": logdir,
        "evaluation_freq": 10,
        "reward_on_episode": False,
        "save_path": save_path,
        "load_path": None,
        "expert_ratio": 0.05,
        "data_path_list": data_path_list,
        "use_rnd": True,
        "rnd_d_model": 64,
        "rnd_depth": 2,
        "rnd_n_heads": 8,
        "rnd_d_key": 8,
        "rnd_d_hidden": 64,
        "rnd_n_omega": 64,
        "rnd_transformer": "preln",
        "rnd_final_ln": True,
        "rnd_use_graph_context": True,
    }
    dqn = TSPDQN(
        **args
    )
    dqn.start()
