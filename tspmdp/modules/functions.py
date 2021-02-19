import inspect
from typing import Callable
import tensorflow as tf


def attention(Q, K, V):
    divide_const = tf.sqrt(tf.cast(tf.constant(K.shape[-1]), tf.float32))
    return tf.matmul(tf.nn.softmax(tf.divide(tf.matmul(Q, K, transpose_b=True), divide_const)), V)


def int_and(x: tf.Tensor, y: tf.Tensor):
    return x * y


def int_or(x: tf.Tensor, y: tf.Tensor):
    return tf.cast(tf.cast(x + y, tf.bool), tf.int32)


def int_not(x: tf.Tensor):
    return (-1) * x + 1


def int_xor(x: tf.Tensor, y: tf.Tensor):
    return tf.math.mod(x + y, 2)


def sample_action(p: tf.Tensor):
    # B, N
    tf.assert_rank(p, 2)
    # B, 1
    return tf.squeeze(tf.random.categorical(tf.math.log(p), 1, dtype=tf.int32), axis=-1)


def get_args(offset: int = None) -> dict:
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args[offset:]}


def cut_off(func: Callable, arguments: dict):
    valid_arguments = inspect.signature(func).parameters
    parameters = {key: value for key,
                  value in arguments.items() if key in valid_arguments}
    return parameters
