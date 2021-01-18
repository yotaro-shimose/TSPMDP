import pytest
import tensorflow as tf

from tspmdp.dqn.actor import SlidingWindowUCB


class DummyRewardProvider:
    def __init__(self, batch_size: int, mean: tf.Tensor):
        self.batch_size = batch_size
        self.mean = mean

    def __call__(self, mode: tf.Tensor):
        """

        Args:
            mode (tf.Tensor): B, M
        """
        mode = tf.cast(mode, tf.float32)
        mean = tf.reduce_sum(tf.expand_dims(self.mean, 0) * mode, -1)
        reward = tf.random.normal(
            shape=(self.batch_size,), mean=mean)
        return reward


@pytest.mark.parametrize("is_beta_zero", [False, True])
def test_swucb(is_beta_zero: bool):
    B, M = 128, 5
    window_size = 1000
    if is_beta_zero:
        beta = 0
    else:
        beta = 100
    eps = 0.01
    assert_eps = 0.5
    iteration = 2000
    reward_provider = DummyRewardProvider(
        B, tf.cast(tf.range(M), tf.float32) * 4)
    ucb = SlidingWindowUCB(
        batch_size=B, n_modes=M, window_size=window_size, beta=beta, eps=eps)
    mean_modes = list()
    for _ in range(iteration):
        reward = reward_provider(ucb.mode)
        mean_modes.append(tf.reduce_mean(
            tf.cast(tf.argmax(ucb.mode, -1), tf.float32)))
        ucb.step(reward)
    if is_beta_zero:
        assert tf.reduce_mean(
            tf.stack(mean_modes[-int(iteration/10):])) >= M - 1 - assert_eps
    else:
        assert int(tf.reduce_mean(tf.stack(mean_modes))) == int((M-1) / 2)
