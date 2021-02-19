import tensorflow as tf


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, maximum_lr: float, warmup_steps: int = 4000):
        super().__init__()

        self.maximum_lr = tf.cast(maximum_lr, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step: tf.Tensor):
        warmup = self.maximum_lr * step / self.warmup_steps
        decay = self.maximum_lr * \
            tf.math.rsqrt(step) / tf.math.rsqrt(self.warmup_steps)
        return tf.where(step < self.warmup_steps, warmup, decay)
