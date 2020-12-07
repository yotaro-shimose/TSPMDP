import tensorflow as tf


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, n_omega=64, eps=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._built_with_F = False
        self.n_omega = n_omega
        self.eps = eps

    def _build_with_F(self, d_feature):
        F = d_feature
        M = self.n_omega
        self.omega: tf.Variable = self.add_weight(
            shape=(M, F),
            trainable=False
        )
        self._built_with_F = True
        self.omega.assign(self._draw_omega(F))

    def _draw_omega(self, d_feature: int):
        n_full_stack = self.n_omega // d_feature
        remainder = self.n_omega - n_full_stack * d_feature
        stack = tf.TensorArray(dtype=tf.float32, size=0,
                               dynamic_size=True, infer_shape=False)
        for i in tf.range(n_full_stack):
            # F, F
            random_features = tf.random.normal((d_feature, d_feature))
            # F, F
            q, _ = tf.linalg.qr(random_features)
            q = q * tf.sqrt(float(d_feature))
            stack = stack.write(i, q)

        random_features = tf.random.normal((d_feature, d_feature))
        # F, F
        q, _ = tf.linalg.qr(random_features)
        q = q * tf.sqrt(float(d_feature))
        stack: tf.TensorArray = stack.write(n_full_stack, q[:remainder])
        # M, F
        kernel = stack.concat()
        stack.close()
        return kernel

    def _phi(self, x: tf.Tensor):
        """

        Args:
            x (tf.Tensor): B, T, F
        """
        # B, T, 1
        square_sum = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) / 2.
        # B, T, M
        xw = tf.einsum("...ik,...jk->...ij", x, self.omega)
        # B, T, 2M
        unnormalized = tf.concat(
            [xw, -xw], axis=-1) - square_sum
        unnormalized = tf.exp(unnormalized) + self.eps
        normalizer = tf.sqrt(2. * self.n_omega)
        normalized = unnormalized / normalizer
        return normalized

    def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
        """

        Args:
            query (tf.Tensor): B, T, F
            value (tf.Tensor): B, S, D
            key (tf.Tensor): B, S, F

        Returns:
            tf.Tensor: B, T, D
        """
        if not self._built_with_F:
            self._build_with_F(value.shape[-1])
        # B, T, M
        query = self._phi(query)
        # B, S, M
        key = self._phi(key)

        ones_shape = value.shape[:-1] + [1, ]

        # B, S, D + 1
        value = tf.concat(
            [value, tf.ones(ones_shape, dtype=tf.float32)], axis=-1)
        # B, M, D + 1
        keyvalue = tf.einsum("...ki,...kj->...ij", key, value)
        # B, T, D + 1
        qkv = tf.einsum("...ik,...kj->...ij", query, keyvalue)
        # B, T, 1
        normalizer = tf.expand_dims(qkv[:, :, -1], -1)
        # B, T, D
        qkv = qkv[:, :, :-1]

        return qkv / normalizer


class MultiHeadLinearAttention(tf.keras.layers.Layer):
    def __init__(self, d_value=128, d_key=128, n_heads=8, n_omega=64, eps=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._built = False
        self.n_omega = n_omega
        self.eps = eps
        self.n_heads = n_heads
        self.d_value = d_value
        self.d_key = d_key
        assert n_omega % n_heads == 0, "n_omega must be multiple of n_heads"
        assert d_key % self.n_heads == 0,  "d_key must be multiple of n_heads"

    def _build(self, d_query, d_value, d_key):
        d_key_per_head = int(self.d_key / self.n_heads)
        d_value_per_head = int(self.d_value / self.n_heads)
        self.wq = self.add_weight(
            name="wq",
            shape=(self.n_heads, d_query, d_key_per_head),
            initializer="glorot_uniform",
            dtype=tf.float32,
            trainable=True,
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(self.n_heads, d_value, d_value_per_head),
            initializer="glorot_uniform",
            dtype=tf.float32,
            trainable=True,
        )
        self.wk = self.add_weight(
            name="wk",
            shape=(self.n_heads, d_key, d_key_per_head),
            initializer="glorot_uniform",
            dtype=tf.float32,
            trainable=True,
        )
        self.omega: tf.Variable = self.add_weight(
            name="omega",
            shape=(self.n_omega, d_key_per_head),
            trainable=False
        )
        self._built = True
        self.omega.assign(self._draw_omega(d_key_per_head))

    def _draw_omega(self, d_feature: int):
        n_full_stack = self.n_omega // d_feature
        remainder = self.n_omega - n_full_stack * d_feature
        stack = tf.TensorArray(dtype=tf.float32, size=0,
                               dynamic_size=True, infer_shape=False)
        for i in tf.range(n_full_stack):
            # F, F
            random_features = tf.random.normal((d_feature, d_feature))
            # F, F
            q, _ = tf.linalg.qr(random_features)
            q = q * tf.sqrt(float(d_feature))
            stack = stack.write(i, q)

        random_features = tf.random.normal((d_feature, d_feature))
        # F, F
        q, _ = tf.linalg.qr(random_features)
        q = q * tf.sqrt(float(d_feature))
        stack: tf.TensorArray = stack.write(n_full_stack, q[:remainder])
        # M, F
        kernel = stack.concat()
        stack.close()
        return kernel

    def _phi(self, x: tf.Tensor):
        """

        Args:
            x (tf.Tensor): B, H, T, F
        """
        # B, H, T, 1
        square_sum = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) / 2.
        # B, H, T, M
        xw = tf.einsum("...ik,...jk->...ij", x, self.omega)
        # B, H, T, 2M
        unnormalized = tf.concat(
            [xw, -xw], axis=-1) - square_sum
        unnormalized = tf.exp(unnormalized) + self.eps
        normalizer = tf.sqrt(2. * self.n_omega)
        normalized = unnormalized / normalizer
        return normalized

    def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
        """

        Args:
            query (tf.Tensor): B, T, F
            value (tf.Tensor): B, S, D
            key (tf.Tensor): B, S, F

        Returns:
            tf.Tensor: B, T, D
        """
        if not self._built:
            self._build(query.shape[-1], value.shape[-1], key.shape[-1])

        # Get shape variables
        B, T, _ = query.shape

        # Linear Projection
        # B, H, T, F'
        query = tf.einsum("...ij,...hjk->...hik", query, self.wq)
        # B, H, S, D'
        value = tf.einsum("...ij,...hjk->...hik", value, self.wv)
        # B, H, S, F'
        key = tf.einsum("...ij,...hjk->...hik", key, self.wk)

        # B, H, T, M
        query = self._phi(query)
        # B, S, M
        key = self._phi(key)

        ones_shape = value.shape[:-1] + [1]

        # B, H, S, D' + 1
        value = tf.concat(
            [value, tf.ones(ones_shape, dtype=tf.float32)], axis=-1)
        # B, H, M, D' + 1
        keyvalue = tf.einsum("...ki,...kj->...ij", key, value)
        # B, H, T, D' + 1
        qkv = tf.einsum("...ik,...kj->...ij", query, keyvalue)
        # B, H, T, 1
        normalizer = tf.expand_dims(qkv[:, :, :, -1], -1)
        # B, H, T, D'
        qkv = qkv[:, :, :, :-1]

        return tf.reshape(qkv / normalizer, (B, T, self.d_value))
