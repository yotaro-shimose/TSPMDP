import numpy as np
from tspmdp.dqn.server import Server


def create_server_args(size, n_nodes, n_step, gamma):
    env_dict = {
        "graph": {"shape": (n_nodes, 2), "dtype": np.float},
        "status": {"shape": (2,), "dtype": np.int},
        "mask": {"shape": (n_nodes,), "dtype": np.int},
        "action": {"shape": (), "dtype": np.int},
        "reward": {"shape": (), "dtype": np.float},
        "next_status": {"shape": (2,), "dtype": np.int},
        "next_mask": {"shape": (n_nodes,), "dtype": np.int},
        "done": {"shape": (), "dtype": np.int},

    }
    Nstep = {"size": n_step,
             "gamma": gamma,
             "rew": "reward",
             "next": ["next_status", "next_mask"]
             }
    return {
        "size": size,
        "env_dict": env_dict,
        "n_step_dict": Nstep
    }


class TSPDQN:
    def __init__(
        self,
        buffer_size=1000000,
        n_nodes=100,
        n_step=3,
        gamma=0.9999
    ):
        # Define server
        args = create_server_args(
            size=buffer_size, n_nodes=n_nodes, n_step=n_step, gamma=gamma)
        self.server = Server(**args)

        # Define actor

        # Define learner

        # Run actor

        # Run learner

        # Run server
        actor

        self,
        server: Server,
        env_builder: Callable,
        network_builder: Callable,
        logger_builder: Callable,
        n_episodes: int = 10000,
        batch_size: int = 128,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        annealing_step: int = 100000,
        data_push_freq: int = 5,
        download_weights_freq: int = 10,
