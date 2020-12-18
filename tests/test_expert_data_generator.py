from tspmdp.dqn.server import Server
from expert_data_generator import create_expert_data, load_expert_data, save_expert_data
import numpy as np
import datetime
import pathlib
import os


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


def test_expert_data_generator():
    n_samples = 5
    buffer_size = 1000
    n_nodes = 100
    n_step = 3
    gamma = 0.999
    factor = 10000
    batch_size = 128
    args = create_server_args(
        size=buffer_size, n_nodes=n_nodes, n_step=n_step, gamma=gamma)
    server = Server(**args)
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    file_path = str(pathlib.Path(r"./expert_data") /
                    f"{n_nodes}nodes{n_samples}samples{date}.dat")
    data = create_expert_data(n_samples, n_nodes, factor)
    save_expert_data(file_path, data)
    loaded_data = load_expert_data(file_path)
    os.remove(file_path)
    server.start()
    server.add(loaded_data)
    sample = server.sample(batch_size)
    assert sample["graph"].shape == (batch_size, n_nodes, 2)
    assert sample["mask"].shape == (batch_size, n_nodes)
    assert sample["status"].shape == (batch_size, 2)
    assert sample["action"].shape == (batch_size, 1)
    assert sample["reward"].shape == (batch_size, 1)
    assert sample["next_mask"].shape == (batch_size, n_nodes)
    assert sample["next_status"].shape == (batch_size, 2)
    assert sample["done"].shape == (batch_size, 1)
    server.terminate()
