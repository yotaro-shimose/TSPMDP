"""
Delivery Markov Decision Process Simulator.
Once instances are created, all of the process(step, reset) must be compiled by tf.function using
integer programming.
"""
import tensorflow as tf
import tree


class TSPMDP(tf.Module):
    """DMDP simulator class.
    Make sure simulator inherit tf.Module so that variables are properly managed.

    """

    def __init__(
            self,
            batch_size: int = 16,
            n_nodes: int = 20,
            seed: int = None
    ):
        # random generator
        if seed is None:
            self.rand_generator = tf.random.Generator.from_non_deterministic_state()
        else:
            self.rand_generator = tf.random.Generator.from_seed(seed)
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
        # B
        self.state_dict = {
            'coordinates': self.coordinates,
            'last_masks': self.last_masks,
            'counts': self.counts,
            'currents': self.currents,
        }

        # Scalars
        self.batch_size = batch_size
        self.n_nodes = n_nodes

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

        # Update counts
        self.counts.assign_add(one_hot_actions)
        # Update current nodes
        self.currents.assign(actions)

        # Calculate reward
        # B
        rewards = cost * (-1)

        # Calculate is_terminals
        # B
        is_terminals = tf.reduce_all(tf.cast(self.counts, tf.bool))

        states = _, _, masks = self.get_states()
        self.last_masks.assign(masks)

        return [states, rewards, is_terminals]

    # @tf.function
    def reset(self):
        self.coordinates.assign(
            self.rand_generator.uniform(self.coordinates.shape))
        # B, N
        self.counts.assign(
            tf.zeros((self.batch_size, self.n_nodes), dtype=tf.int32))
        # B
        self.depos.assign(self._init_depos())
        # B
        self.currents.assign(tf.identity(self.depos))

        # [(B, N, 2), (B, 2), B]
        states = _, _, masks = self.get_states()
        self.last_masks.assign(masks)
        return states

    def get_states(self):
        # B, N
        masks = self._get_mask()

        # B, N, 2
        graphs = self.coordinates

        # B, 3
        status = self._get_status()

        return [graphs, status, masks]

    def _get_status(self):
        # B, 3
        status = tf.stack([self.currents, self.depos], axis=-1)
        return status

    def _get_mask(self):
        # B, N
        only_once = 1 - tf.cast(self.counts ==
                                tf.ones(self.counts.shape, dtype=tf.int32), tf.int32)
        # B, N
        never_stay = 1 - tf.one_hot(
            self.currents, depth=self.n_nodes, dtype=tf.int32)
        # B, N
        one_hot_depo = tf.one_hot(
            self.depos, depth=self.n_nodes, dtype=tf.int32)
        # B, 1
        finished_clients = tf.cast(tf.reduce_all(
            (one_hot_depo +
             self.counts) == tf.ones(self.counts.shape, dtype=tf.int32),
            axis=-1, keepdims=True), tf.int32)
        depo_last = 1 - (1 - finished_clients) * one_hot_depo
        return only_once * never_stay * depo_last

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
        return tree.map_structure(tf.identity, self.state_dict)