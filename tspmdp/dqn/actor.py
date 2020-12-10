import tensorflow as tf


class Actor:
    def start(self):
        for episode in range(self.n_episodes):
            metrics = self._episode()
            self.logger.log_metrics(metrics)2

    @tf.function
    def _act(self, state: tf.Tensor):
        action = self._get_action(state)
        next_state, reward, done = self.env.step(action)
        return action, reward, next_state, done

    def _episode(self):
        state = self.env.reset()
        done, episode_reward, ones = self._init_episode()
        while tf.math.logical_not(tf.reduce_all(done == ones)):
            action, reward, next_state, done = self._act(state)
            self._memorize(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            self._on_step_end()
        metrics = {"episode_reward": episode_reward}
        return metrics

    def _memorize(self, next_state, state):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError

    def _init_episode(self):
        shape = tf.TensorSpec(shape=[self.batch_size])
        done = tf.zeros(shape, dtype=tf.int32)
        episode_reward = tf.zeros(shape, dtype=tf.float32)
        ones = tf.ones(shape, dtype=tf.int32)
        return done, episode_reward, ones

    def _on_step_end(self):
        # Flush the memory

        # Download parameters

        # Anneal epsilon
        raise NotImplementedError

    def _get_action(self, state):
        q_values = self.network(state)

        greedy_action = tf.argmax(q_values, axis=1, output_type=tf.int32)
        random_action = self._randint(
            greedy_action.shape, min=0, max=tf.size(q_values))
        random_flag = tf.cast(tf.random.uniform(
            shape=greedy_action.shape) < self.eps, tf.int32)
        return random_flag * random_action + (1. - random_flag) * greedy_action

    def _randint(self, shape: tf.TensorShape, min: int, max: int) -> tf.Tensor:
        uniform = tf.random.uniform(
            shape=shape, minval=min, maxval=max)
        return tf.cast(tf.floor(uniform), tf.int32)

    def _initialize(self):
        # Build network instance
        # Build network weights

        raise NotImplementedError
