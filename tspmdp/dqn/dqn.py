from multiprocessing import Process
from typing import List, Union

import numpy as np
from tspmdp.dqn.actor import Actor
from tspmdp.dqn.learner import Learner
from tspmdp.dqn.server import ReplayBuffer, Server
from tspmdp.env import TSPMDP
from tspmdp.expert_data_generator import load_expert_data
from tspmdp.logger import TFLogger
from tspmdp.modules.functions import get_args
from tspmdp.modules.rnd import RNDBuilder
from tspmdp.network_builder import CustomizableNetworkBuilder

NON_DISPLAY_HPARAMS = ["data_path_list", "save_path", "load_path", "logdir"]


def create_server_args(size, n_nodes, n_step, gamma):
    env_dict = {
        "graph": {"shape": (n_nodes, 2), "dtype": np.float32},
        "status": {"shape": (2,), "dtype": np.int32},
        "mask": {"shape": (n_nodes,), "dtype": np.int32},
        "action": {"shape": (1,), "dtype": np.int32},
        "reward": {"dtype": np.float32},
        "next_status": {"shape": (2,), "dtype": np.int32},
        "next_mask": {"shape": (n_nodes,), "dtype": np.int32},
        "done": {"shape": (1,), "dtype": np.int32},
    }
    # Nstep = {"size": n_step,
    #          "gamma": gamma,
    #          "rew": "reward",
    #          "next": ["next_status", "next_mask", "done"]
    #          }
    return {
        "size": size,
        "env_dict": env_dict,
        # "n_step_dict": Nstep
    }


class EnvBuilder:
    def __init__(self, batch_size, n_nodes, reward_on_episode):
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.reward_on_episode = reward_on_episode

    def __call__(self):
        return TSPMDP(
            batch_size=self.batch_size,
            n_nodes=self.n_nodes,
            reward_on_episode=self.reward_on_episode
        )


class LoggerBuilder:
    def __init__(self, logdir, hparams: dict = None):
        self.logdir = logdir
        self.hparams = hparams

    def __call__(self):
        return TFLogger(self.logdir, self.hparams)


class ReplayBufferBuilder:
    def __init__(self, **args):
        self.args = args

    def __call__(self):
        return ReplayBuffer(**self.args)


class ExpertDataLoader:
    def __init__(self, data_path_list: List[str], reward_on_episode=False):
        self.data_path_list = data_path_list
        if reward_on_episode:
            raise NotImplementedError

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
        gamma: Union[float, list] = 0.9999,
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
        reward_on_episode: bool = True,
        save_path: str = None,
        load_path: str = None,
        n_learner_epochs: int = 1000000,
        learner_batch_size: int = 128,
        learning_rate: float = 1e-3,
        upload_freq: int = 100,
        sync_freq: int = 50,
        scale_value_function: bool = True,
        expert_ratio: float = 0,
        data_path_list: List[str] = None,
        use_rnd: bool = True,
        rnd_d_model: int = 64,
        rnd_depth: int = 2,
        rnd_n_heads: int = 8,
        rnd_d_key: int = 8,
        rnd_d_hidden: int = 64,
        rnd_n_omega: int = 64,
        rnd_transformer: str = "preln",
        rnd_final_ln: bool = True,
        rnd_use_graph_context: bool = True,
        beta: List[float] = [0., 0.2, 0.4, 0.6, 0.8, 1],
        ucb_window_size: int = 16*50,
        ucb_eps: float = 0.5,
        ucb_beta: float = 1,
    ):
        hparams = get_args(offset=1)
        for key in NON_DISPLAY_HPARAMS:
            hparams.pop(key)

        if logdir:
            logger_builder = LoggerBuilder(logdir, hparams)
        else:
            logger_builder = None

        # Define server
        args = create_server_args(
            size=buffer_size, n_nodes=n_nodes, n_step=n_step, gamma=gamma)
        self.server = Server(**args)

        # Define expert data loader
        if expert_ratio > 0:
            replay_buffer_builder = ReplayBufferBuilder(**args)
            data_generator = ExpertDataLoader(
                data_path_list, reward_on_episode=reward_on_episode)
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
        if use_rnd:
            rnd_builder = RNDBuilder(
                d_model=rnd_d_model,
                depth=rnd_depth,
                n_heads=rnd_n_heads,
                d_key=rnd_d_key,
                d_hidden=rnd_d_hidden,
                n_omega=rnd_n_omega,
                transformer=rnd_transformer,
                final_ln=rnd_final_ln,
                use_graph_context=rnd_use_graph_context
            )
        else:
            rnd_builder = None

        # Define env_builder
        env_builder = EnvBuilder(
            batch_size=n_parallels, n_nodes=n_nodes, reward_on_episode=reward_on_episode)
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
            save_path=save_path,
            load_path=load_path,
            beta=beta,
            gamma=gamma,
            ucb_window_size=ucb_window_size,
            ucb_eps=ucb_eps,
            ucb_beta=ucb_beta
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
            rnd_builder=rnd_builder,
            beta=beta,
        )
        self.learner = Process(target=self.learner.start)

    def start(self):
        # Run actor
        self.actor.start()
        # Run learner
        self.learner.start()
        # Run server
        self.server.run()
