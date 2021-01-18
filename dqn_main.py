import datetime
import pathlib
import multiprocessing
from tspmdp.dqn.dqn import TSPDQN


# TODO
# - make actor to choose exploration mode using sliding UCB
# - parametrize to switch RND

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", True)
    n_nodes = 100
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    logdir = str(pathlib.Path("./logs/") / "DQN_EXP" /
                 ("nodes" + str(n_nodes)) / date)
    data_path_list = [
        "./expert_data/100nodes1000samples20201218142925.dat",
        "./expert_data/100nodes1000samples20201218141310.dat",
        "./expert_data/100nodes1000samples20201218135738.dat",
    ]

    save_path = None
    use_rnd = True
    base = {
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
        "buffer_size": 1000000,
        "eps_start": 0.,
        "eps_end": 0.,
        "annealing_step": 100000,
        "data_push_freq": 5,
        "download_weights_freq": 5,
        "n_learner_epochs": 1000000,
        "learner_batch_size": 256,
        "learning_rate": 1e-3,
        "upload_freq": 100,
        "sync_freq": 100,
        "scale_value_function": False,
        "logdir": logdir,
        "evaluation_freq": 10,
        "reward_on_episode": False,
        "save_path": save_path,
        "load_path": None,
        "expert_ratio": 0.,
        "data_path_list": data_path_list,
        "use_rnd": use_rnd,

    }

    rnd_args = {
        "rnd_d_model": 64,
        "rnd_depth": 2,
        "rnd_n_heads": 8,
        "rnd_d_key": 8,
        "rnd_d_hidden": 64,
        "rnd_n_omega": 64,
        "rnd_transformer": "preln",
        "rnd_final_ln": True,
        "rnd_use_graph_context": True,
        "ucb_window_size": 16*50,
        "ucb_eps": 0.5,
        "ucb_beta": 1.,
        "beta": [0., 0.1, 0.2, 0.25, 0.5, 1.0],
        "gamma": [0.999, 0.999, 0.999, 0.999, 0.999, 0.999],
    }
    if use_rnd:
        base.update(rnd_args)

    dqn = TSPDQN(
        **base
    )
    dqn.start()
