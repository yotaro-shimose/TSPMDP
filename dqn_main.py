import datetime
import pathlib

from tspmdp.dqn.dqn import TSPDQN

if __name__ == '__main__':
    n_parallels = 128
    n_nodes = 100
    n_episodes = 100000
    n_step = 3
    gamma = 0.9999
    d_model: int = 128
    depth: int = 6
    n_heads: int = 8
    d_key: int = 16
    d_hidden: int = 128
    n_omega: int = 64
    transformer: str = "preln"
    final_ln: bool = True
    decoder_mha: str = "softmax"
    use_graph_context: bool = True
    buffer_size = 1000000
    eps_start: float = 1.0
    eps_end: float = 0.01
    annealing_step: int = 100000
    data_push_freq: int = 1
    download_weights_freq: int = 10
    n_learner_epochs: int = 1000000
    learner_batch_size: int = 512
    learning_rate: float = 1e-3
    upload_freq: int = 100
    sync_freq: int = 50
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    logdir = str(pathlib.Path("./logs/") / "DQN" /
                 ("nodes" + str(n_nodes)) / date)
    dqn = TSPDQN(
        n_parallels=n_parallels,
        n_nodes=n_nodes,
        n_episodes=n_episodes,
        n_step=n_step,
        gamma=gamma,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_key=d_key,
        d_hidden=d_hidden,
        n_omega=n_omega,
        transformer=transformer,
        final_ln=final_ln,
        decoder_mha=decoder_mha,
        use_graph_context=use_graph_context,
        logdir=logdir,
        buffer_size=buffer_size,
        eps_start=eps_start,
        eps_end=eps_end,
        annealing_step=annealing_step,
        data_push_freq=data_push_freq,
        download_weights_freq=download_weights_freq,
        n_learner_epochs=n_learner_epochs,
        learner_batch_size=learner_batch_size,
        learning_rate=learning_rate,
        upload_freq=upload_freq,
        sync_freq=sync_freq,
    )
    dqn.start()
