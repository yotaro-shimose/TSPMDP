import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class TFLogger:

    def __init__(self, logdir, hparams: dict = None):
        self.writer = tf.summary.create_file_writer(logdir=logdir)
        self.hparams = hparams

    def log(self, metrics: dict, step: int):
        with self.writer.as_default():
            if self.hparams:
                hp.hparams(self.hparams)
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step)


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
