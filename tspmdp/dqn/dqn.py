from multiprocessing import Process
from typing import List

import numpy as np
from tspmdp.dqn.actor import Actor
from tspmdp.dqn.learner import Learner
from tspmdp.dqn.server import Server, ReplayBuffer
from tspmdp.env import TSPMDP
from tspmdp.logger import TFLogger
from tspmdp.network_builder import CustomizableNetworkBuilder
from tspmdp.expert_data_generator import load_expert_data


def create_server_args(size, n_nodes, n_step, gamma):
    env_dict = {
        "graph": {"shape": (n_nodes, 2), "dtype": np.float32},
        "status": {"shape": (2,), "dtype": np.int32},
        "mask": {"shape": (n_nodes,), "dtype": np.int32},
        "action": {"dtype": np.int32},
        "reward": {"dtype": np.float32},
        "next_status": {"shape": (2,), "dtype": np.int32},
        "next_mask": {"shape": (n_nodes,), "dtype": np.int32},
        "done": {"dtype": np.int32},
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


class EnvBuilder:
    def __init__(self, batch_size, n_nodes):
        self.batch_size = batch_size
        self.n_nodes = n_nodes

    def __call__(self):
        return TSPMDP(
            batch_size=self.batch_size,
            n_nodes=self.n_nodes
        )


class LoggerBuilder:
    def __init__(self, logdir):
        self.logdir = logdir

    def __call__(self):
        return TFLogger(self.logdir)


class ReplayBufferBuilder:
    def __init__(self, **args):
        self.args = args

    def __call__(self):
        return ReplayBuffer(**self.args)


class ExpertDataLoader:
    def __init__(self, data_path_list: List[str]):
        self.data_path_list = data_path_list

    def __call__(self):
        data = []
        for path in self.data_path_list:
            data += load_expert_data(path)
        return data


class TSPDQN:
    def __init__(
        self,
        n_parallels=128,
        n_nodes=100,
        n_episodes=100000,
        n_step=3,
        gamma=0.9999,
        d_model: int = 128,
        depth: int = 6,
        n_heads: int = 8,
        d_key: int = 16,
        d_hidden: int = 128,
        n_omega: int = 64,
        transformer: str = "preln",
        final_ln: bool = True,
        decoder_mha: str = "softmax",
        use_graph_context: bool = True,
        logdir: str = None,
        buffer_size=1000000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        annealing_step: int = 100000,
        data_push_freq: int = 5,
        download_weights_freq: int = 10,
        evaluation_freq: int = 1000,
        n_learner_epochs: int = 1000000,
        learner_batch_size: int = 128,
        learning_rate: float = 1e-3,
        upload_freq: int = 100,
        sync_freq: int = 50,
        scale_value_function: bool = True,
        expert_ratio: float = 0,
        data_path_list: List[str] = None,
    ):
        if logdir:
            logger_builder = LoggerBuilder(logdir)
        else:
            logger_builder = None

        # Define server
        args = create_server_args(
            size=buffer_size, n_nodes=n_nodes, n_step=n_step, gamma=gamma)
        self.server = Server(**args)

        # Define expert data loader
        if expert_ratio > 0:
            replay_buffer_builder = ReplayBufferBuilder(**args)
            data_generator = ExpertDataLoader(data_path_list)
        else:
            replay_buffer_builder = None
            data_generator = None

        # Define network builder
        network_builder = CustomizableNetworkBuilder(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            d_key=d_key,
            d_hidden=d_hidden,
            n_omega=n_omega,
            transformer=transformer,
            final_ln=final_ln,
            decoder_mha=decoder_mha,
            use_graph_context=use_graph_context
        )

        # Define env_builder
        env_builder = EnvBuilder(batch_size=n_parallels, n_nodes=n_nodes)
        # Define actor
        self.actor = Actor(
            server=self.server,
            env_builder=env_builder,
            network_builder=network_builder,
            logger_builder=logger_builder,
            n_episodes=n_episodes,
            batch_size=n_parallels,
            eps_start=eps_start,
            eps_end=eps_end,
            annealing_step=annealing_step,
            data_push_freq=data_push_freq,
            download_weights_freq=download_weights_freq,
            evaluation_freq=evaluation_freq,
        )
        self.actor = Process(target=self.actor.start)
        # Define learner
        self.learner = Learner(
            server=self.server,
            network_builder=network_builder,
            logger_builder=logger_builder,
            n_epochs=n_learner_epochs,
            batch_size=learner_batch_size,
            learning_rate=learning_rate,
            n_step=n_step,
            gamma=gamma,
            upload_freq=upload_freq,
            sync_freq=sync_freq,
            scale_value_function=scale_value_function,
            expert_ratio=expert_ratio,
            replay_buffer_builder=replay_buffer_builder,
            data_generator=data_generator,
        )
        self.learner = Process(target=self.learner.start)

    def start(self):
        # Run actor
        self.actor.start()
        # Run learner
        self.learner.start()
        # Run server
        self.server.run()
