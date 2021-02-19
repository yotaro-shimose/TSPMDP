"""
Delivery Markov Decision Process Simulator.
Once instances are created, all of the process(step, reset) must be compiled by tf.function using
integer programming.
"""
import tensorflow as tf
from typing import List


class TSPMDP(tf.Module):
    """DMDP simulator class.
    Make sure simulator inherit tf.Module so that variables are properly managed.

    """

    def __init__(
            self,
            batch_size: int = 14,
            n_nodes: int = 20,
            reward_on_episode: bool = False
    ):

        # 1 for depo
        # Per instance variables
        # B, N, 2 (2 for x, y)
        self.coordinates = tf.Variable(
            tf.zeros((batch_size, n_nodes, 2), dtype=tf.float32))
        # B, N
        self.last_masks = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))

        # Per step variables
        self.counts = tf.Variable(
            tf.zeros((batch_size, n_nodes), dtype=tf.int32))
        # STATUS
        # B
        self.currents = tf.Variable(tf.zeros((batch_size,), dtype=tf.int32))
        # B
        self.depos = tf.Variable(-tf.ones((batch_size,), dtype=tf.int32))
        self.state_dict = {
            'coordinates': self.coordinates,
            'last_masks': self.last_masks,
            'counts': self.counts,
            'currents': self.currents,
            'depos': self.depos
        }
        if reward_on_episode:
            # B
            self.rewards = tf.Variable(
                tf.zeros((batch_size,), dtype=tf.float32))
            self.state_dict.update({'rewards': self.rewards})

        # Scalars
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.reward_on_episode = reward_on_episode

    @tf.function
    def step(self, actions: tf.Tensor):
        """step function

        Args:
            actions (tf.Tensor): B
        """
        # B, N
        one_hot_actions = tf.one_hot(
            actions, depth=self.n_nodes, dtype=tf.int32)

        # filter actions by mask, which should not do anything
        # if actions are valid (unless it's done).
        # B, N
        filtered_actions = self.last_masks * one_hot_actions
        non_filtered_actions = one_hot_actions

        # Validate actions
        tf.assert_equal(filtered_actions, non_filtered_actions)

        # Calculate cost
        # B
        cost = tf.norm(self._get_cord(actions) -
                       self._get_cord(self.currents), axis=1)

        closing_cost = tf.norm(self._get_cord(
            self.depos) - self._get_cord(actions), axis=1)

        # Update counts
        self.counts.assign_add(one_hot_actions)
        # Update current nodes
        self.currents.assign(actions)

        # Calculate is_terminals
        # B
        is_terminals = tf.cast(tf.reduce_all(
            tf.cast(self.counts, tf.bool), axis=-1), tf.int32)

        # Calculate reward
        # B
        rewards = (-1) * (cost + tf.cast(is_terminals,
                                         tf.float32) * closing_cost)

        if self.reward_on_episode:
            # B Update Rewards to 0 if not terminal else total rewards.
            self.rewards.assign_add(rewards)
            if tf.reduce_all(tf.cast(is_terminals, tf.bool)):
                rewards = tf.identity(self.rewards)
            else:
                rewards = tf.zeros((self.batch_size,), tf.float32)

        # [(B, N, 2), B, B]
        states = _, _, masks = self.get_states()
        self.last_masks.assign(masks)

        return [states, rewards, is_terminals]

    @tf.function
    def reset(self):
        self.coordinates.assign(
            tf.random.uniform(self.coordinates.shape))
        # B
        self.depos.assign(self._init_depos())
        # B, N
        self.counts.assign(
            tf.one_hot(self.depos, depth=self.n_nodes, dtype=tf.int32)
        )
        # B
        self.currents.assign(tf.identity(self.depos))

        if self.reward_on_episode:
            # B
            self.rewards.assign(tf.zeros((self.batch_size,), dtype=tf.float32))

        # [(B, N, 2), (B, 2), (B, N)]
        states = _, _, masks = self.get_states()
        self.last_masks.assign(masks)
        return states

    def get_states(self) -> List[tf.Tensor]:
        # B, N
        masks = self._get_mask()

        # B, N, 2
        graphs = self.coordinates

        # B, 3
        status = self._get_status()

        return [graphs, status, masks]

    def _get_status(self):
        # B, 2
        status = tf.stack([self.currents, self.depos], axis=-1)
        return tf.identity(status)

    def _get_mask(self):
        # B, N
        only_once = 1 - tf.cast(self.counts ==
                                tf.ones(self.counts.shape, dtype=tf.int32), tf.int32)

        return only_once

    def _init_depos(self):
        return tf.zeros((self.batch_size,), dtype=tf.int32)

    def _get_cord(self, indices: tf.Tensor):
        """get coordinates corresponding to indices

        Args:
            indices (tf.Tensor): B
        """
        indices = tf.one_hot(indices, depth=self.n_nodes)
        indices = tf.stack([indices, indices], axis=2)
        coordinates = tf.reduce_sum(indices * self.coordinates, axis=1)
        return coordinates

    def import_states(
        self,
        states: dict
    ):
        for key, value in states.items():
            self.state_dict[key].assign(value)

    def export_states(self):
        return tf.nest.map_structure(tf.identity, self.state_dict)
