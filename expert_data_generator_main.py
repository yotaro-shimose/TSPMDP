from tspmdp.expert_data_generator import create_expert_data, save_expert_data
import datetime
import pathlib


if __name__ == '__main__':
    n_samples = 1000
    n_nodes = 100
    for _ in range(100):
        date = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        file_path = str(pathlib.Path(r"./expert_data") /
                        f"{n_nodes}nodes{n_samples}samples{date}.dat")
        data = create_expert_data(n_samples, n_nodes)
        save_expert_data(file_path, data)
