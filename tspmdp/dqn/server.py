from multiprocessing import Process, Queue, Pipe, Lock
from cpprb import ReplayBuffer as CPPRB
from typing import Sequence, Dict, Any
import numpy as np


class Server(Process):
    def __init__(self, size, env_dict, n_step_dict=None, min_storage=10000, done_string="done"):
        super().__init__()
        self.done_string = done_string
        self.queue = Queue()
        self.size = size
        self.client_pipe, self.server_pipe = Pipe()
        self.env_dict = env_dict
        self.n_step_dict = n_step_dict
        self.parameter = None
        self.min_storage = min_storage
        self.cpprb_args = {
            "size": size,
            "env_dict": env_dict,
            "Nstep": n_step_dict
        }

        # Server lock object
        self.lock = Lock()

    def run(self) -> None:
        self.buffer = CPPRB(
            **self.cpprb_args)
        while True:
            cmd, *args = self.queue.get()
            if cmd == "add":
                self._add(*args)
            elif cmd == "sample":
                self.server_pipe.send(self._sample(*args))
            elif cmd == "upload":
                self._upload(*args)
            elif cmd == "download":
                self.server_pipe.send(self._download())
            else:
                raise ValueError(
                    f"Parameter Server got an unexpected command {cmd}")

    def _download(self) -> Any:
        return self.parameter

    def _upload(self, parameter: Any) -> None:
        self.parameter = parameter

    def _add(self, data: Dict[str, Sequence[np.ndarray]]) -> None:
        self.buffer.add(**data)

    def _sample(self, size: int) -> Dict[str, np.ndarray]:
        if self.buffer.get_stored_size() < self.min_storage:
            print(
                f"stored sample {self.buffer.get_stored_size()} is smaller than mininum storage\
                     size {self.min_storage}. Returning None.")
            return None
        else:
            return self.buffer.sample(size)

    def download(self) -> Any:
        cmd = "download"
        self.lock.acquire()
        self.queue.put((cmd, None))
        weights = self.client_pipe.recv()
        self.lock.release()
        return weights

    def upload(self, parameter: Any):
        cmd = "upload"
        self.queue.put((cmd, parameter))

    def add(self, data: Sequence[Dict[str, np.ndarray]]):
        cmd = "add"
        self.queue.put((cmd, data))

    def sample(self, size: int) -> Dict[str, np.ndarray]:
        cmd = "sample"
        self.lock.acquire()
        self.queue.put((cmd, size))
        sample = self.client_pipe.recv()
        self.lock.release()
        return sample


class ReplayBuffer:
    def __init__(self, size, env_dict, n_step_dict=None, min_storage=10000, done_string="done"):
        super().__init__()
        self.done_string = done_string
        self.min_storage = min_storage
        cpprb_args = {
            "size": size,
            "env_dict": env_dict,
            "Nstep": n_step_dict
        }
        self.buffer = CPPRB(**cpprb_args)

    def add(self, data: Sequence[Dict[str, np.ndarray]]) -> None:
        for d in data:
            self.buffer.add(**d)
            if d[self.done_string]:
                self.buffer.on_episode_end()

    def sample(self, size: int) -> Dict[str, np.ndarray]:
        if self.buffer.get_stored_size() < self.min_storage:
            print(
                f"stored sample {self.buffer.get_stored_size()} is smaller than mininum storage" +
                f"size {self.min_storage}. Returning None."
            )
            return None
        else:
            return self.buffer.sample(size)
