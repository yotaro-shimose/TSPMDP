
import tensorflow as tf
from typing import List


def distance_array(kernel: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """compute distance

    Args:
        x (tf.Tensor): B, 2
        y (tf.Tensor): B, N, 2

    Returns:
        tf.Tensor: B
    """
    assert kernel.dtype == tf.float32
    assert targets.dtype == tf.float32
    # B, 1, 2
    kernel = tf.expand_dims(kernel, axis=-2)
    # B, N, 2
    dif = targets - kernel
    return tf.sqrt(tf.reduce_sum(tf.square(dif), axis=-1))


class DummyEnv(tf.Module):

    def __init__(
            self,
            batch_size: int = 14,
            n_nodes: int = 20,
            *args,
            **kwargs,
    ):

        # B, N, 2 (2 for x, y)
        self.coordinates = tf.Variable(
            tf.zeros((batch_size, n_nodes, 2), dtype=tf.float32))

        # B, N
        self.counts = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))

        # B, N
        self.kernel = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))

        # B, N
        self.distances = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.float32))

        # B, N
        self.last_mask = tf.Variable(
            tf.identity(self.kernel)
        )

        # TODO
        self.state_dict = {
            "coordinates": self.coordinates,
            "counts": self.counts,
            "kernel": self.kernel,
            "distances": self.distances
        }

        # Scalars
        self.batch_size = batch_size
        self.n_nodes = n_nodes

    # @tf.function
    def step(self, action):
        """step function

        Args:
            action (tf.Tensor): B
        """
        # B, N
        one_hot_actions = tf.one_hot(
            action, depth=self.n_nodes, dtype=tf.int32)

        # filter actions by mask, which should not do anything
        # if actions are valid (unless it's done).
        # B, N
        filtered_actions = self.last_mask * one_hot_actions
        non_filtered_actions = one_hot_actions

        # Validate actions
        tf.assert_equal(filtered_actions, non_filtered_actions)

        # Calculate cost
        # B, 1
        distance = tf.reduce_sum(
            tf.cast(one_hot_actions, tf.float32) * self.distances, keepdims=True, axis=-1)
        # B, N
        rank = tf.cast(distance > self.distances, tf.int32) * self.last_mask
        # B
        reward = tf.cast(tf.reduce_sum(rank, axis=-1), tf.float32)

        # Update counts
        self.counts.assign_add(one_hot_actions)
        # Calculate is_terminals
        # B
        is_terminals = tf.cast(tf.reduce_all(
            tf.cast(self.counts, tf.bool), axis=-1), tf.int32)

        # [(B, N, 1), B, B]
        states = _, _, masks = self.get_states()
        self.last_mask.assign(masks)

        return [states, reward, is_terminals]

    @tf.function
    def reset(self):
        self.coordinates.assign(
            tf.random.uniform(self.coordinates.shape, minval=-1, maxval=1))
        # B, N
        self.kernel.assign(
            tf.one_hot(tf.zeros((self.batch_size,), dtype=tf.int32),
                       depth=self.n_nodes, dtype=tf.int32)
        )
        self.counts.assign(
            tf.identity(self.kernel)
        )
        # B, 2
        kernel_coordinate = tf.reduce_sum(tf.cast(tf.expand_dims(
            self.kernel, axis=-1), tf.float32) * self.coordinates, axis=-2)
        # B
        self.distances.assign(
            distance_array(kernel_coordinate, self.coordinates)
        )

        states = _, _, mask = self.get_states()
        # [(B, N, 2), (B, 2), (B, N)]
        self.last_mask.assign(mask)
        return states

    def get_states(self) -> List[tf.Tensor]:
        # B, N
        masks = self._get_mask()

        # B, N, 2
        graphs = tf.identity(self.coordinates)

        # B, 3
        status = self._get_status()

        return [graphs, status, masks]

    def _get_status(self):
        # B, 1
        status = tf.expand_dims(self.kernel, axis=-1)
        return status

    def _get_mask(self):
        # B, N
        only_once = 1 - self.counts

        return only_once

    def import_states(
        self,
        states: dict
    ):
        for key, value in states.items():
            self.state_dict[key].assign(value)

    def export_states(self):
        return tf.nest.map_structure(tf.identity, self.state_dict)
