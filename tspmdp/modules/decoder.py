import tensorflow as tf
from typing import List
from tspmdp.modules.source_generator import SourceGenerator, WouterSourceGenerator
from tspmdp.modules.functions import cut_off, get_args


MHA_TYPES = {
    "softmax": tf.keras.layers.MultiHeadAttention,
    # "linear": LinearAttention,
    "none": None
}


@tf.keras.utils.register_keras_serializable()
class CustomizableQDecoder(tf.keras.models.Model):
    def __init__(
        self,
        n_heads: int,
        d_key: int,
        th_range: float,
        n_omega: int = 64,
        mha: str = "softmax",
        use_graph_context: bool = True,
        *args,
        **kwargs
    ):
        # Make sure valid transformer type is specified
        assert mha in MHA_TYPES, \
            f"argument 'mha' should be one of {str(list(MHA_TYPES.keys()))}"
        super().__init__(*args, **kwargs)

        # save init arguments
        self.init_args = get_args(offset=1)
        self.d_key = d_key
        self.th_range = th_range
        self.source_generator = SourceGenerator(
            use_graph_context=use_graph_context)
        MHAClass = MHA_TYPES[mha]
        mha_args = {
            "n_heads": n_heads,
            "num_heads": n_heads,
            "key_dim": d_key,
            "d_key": d_key,
            "n_omega": n_omega
        }
        mha_args = cut_off(MHAClass.__init__, mha_args)
        if mha == "none":
            self.mha = None
        else:
            self.mha = MHAClass(**mha_args)

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        initializer = tf.keras.initializers.GlorotNormal()
        self.wq = self.add_weight(name="wq", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [H(B, N, D), indice(B, F), mask(B, N)]
        """
        H = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        # B, 1, D
        query = self.source_generator([H, indice])

        # B, 1, N
        mask = tf.expand_dims(mask, axis=1)
        # B, 1, D
        if self.mha:
            query = self.mha(query, H, H, mask)

        # B, 1, D
        q_value = self._calc_Q(query, H, mask)
        return q_value

    def _calc_Q(self, query: tf.Tensor, H: tf.Tensor, mask: tf.Tensor):
        """calculate policy based on query, graph_embedding and mask

        Args:
            query (tf.Tensor): B, 1, D
            H (tf.Tensor): B, N, D
            mask (tf.Tensor): B, 1, D

        Returns:
            [type]: [description]
        """
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(H, self.wk)
        scale = tf.sqrt(float(self.d_key))
        # B, 1, N
        QK = tf.matmul(Q, K, transpose_b=True) / scale
        # B, 1, N
        mask = tf.cast(mask, tf.float32)
        mask = (1 - mask) * 0
        q_value = QK * mask
        # now q_value is tensor of shape(B, 1, N) which must be turned into tensor of
        # shape(B, N)
        return tf.squeeze(q_value, axis=1)

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base


@tf.keras.utils.register_keras_serializable()
class CustomizablePolicyDecoder(tf.keras.models.Model):
    def __init__(
        self,
        n_heads: int,
        d_key: int,
        th_range: float,
        n_omega: int = 64,
        mha: str = "softmax",
        use_graph_context: bool = True,
        *args,
        **kwargs
    ):
        # Make sure valid transformer type is specified
        assert mha in MHA_TYPES, \
            f"argument 'mha' should be one of {str(list(MHA_TYPES.keys()))}"
        super().__init__(*args, **kwargs)

        # save init arguments
        self.init_args = get_args(offset=1)
        self.d_key = d_key
        self.th_range = th_range
        self.source_generator = SourceGenerator(
            use_graph_context=use_graph_context)
        MHAClass = MHA_TYPES[mha]
        mha_args = {
            "n_heads": n_heads,
            "num_heads": n_heads,
            "key_dim": d_key,
            "d_key": d_key,
            "n_omega": n_omega
        }
        mha_args = cut_off(MHAClass.__init__, mha_args)
        if mha == "none":
            self.mha = None
        else:
            self.mha = MHAClass(**mha_args)

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        initializer = tf.keras.initializers.GlorotNormal()
        self.wq = self.add_weight(name="wq", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [H(B, N, D), indice(B, F), mask(B, N)]
        """
        H = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        # B, 1, D
        query = self.source_generator([H, indice])

        # B, 1, N
        mask = tf.expand_dims(mask, axis=1)
        # B, 1, D
        if self.mha:
            query = self.mha(query, H, H, mask)

        # B, 1, D
        policy = self._calc_policy(query, H, mask)
        return policy

    def _calc_policy(self, query: tf.Tensor, H: tf.Tensor, mask: tf.Tensor):
        """calculate policy based on query, graph_embedding and mask

        Args:
            query (tf.Tensor): B, 1, D
            H (tf.Tensor): B, N, D
            mask (tf.Tensor): B, 1, D

        Returns:
            [type]: [description]
        """
        INFINITE = 1e+9
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(H, self.wk)
        scale = tf.sqrt(float(self.d_key))
        # B, 1, N
        QK = tf.matmul(Q, K, transpose_b=True) / scale
        # B, 1, N
        mask = tf.cast(mask, tf.float32)
        mask = (1 - mask) * (-INFINITE)
        policy = tf.keras.activations.softmax(
            self.th_range * tf.keras.activations.tanh(QK) + mask)
        # now policy is tensor of shape(B, 1, N) which must be turned into tensor of
        # shape(B, N)
        return tf.squeeze(policy, axis=1)

    def get_config(self) -> dict:
        base: dict = super().get_config()
        base.update(self.init_args)
        return base


class PolicyDecoder(tf.keras.models.Model):

    def __init__(self, d_key, th_range):
        super().__init__()
        self.d_key = d_key
        self.th_range = th_range
        self.source_generator = SourceGenerator()

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        initializer = tf.keras.initializers.GlorotNormal()
        self.wq = self.add_weight(name="wq", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)
        super().build(input_shape)

    def _calc_policy(self, query, H, mask):
        INFINITE = 1e+9
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(H, self.wk)
        scale = tf.sqrt(float(self.d_key))
        # B, 1, N
        QK = tf.matmul(Q, K, transpose_b=True) / scale
        # mask is tensor of shape (B, N) by default.
        # but it must be tensor of shape (B, 1, N).
        batch_size = Q.shape[0]
        n_query = Q.shape[1]  # one
        n_nodes = K.shape[1]
        mask = tf.cast(tf.reshape(
            mask, (batch_size, n_query, n_nodes)), tf.float32)
        mask = (1 - mask) * (-INFINITE)
        policy = tf.keras.activations.softmax(
            self.th_range * tf.keras.activations.tanh(QK) + mask)
        # now policy is tensor of shape(B, 1, N) which must be turned into tensor of
        # shape(B, N)
        return tf.reshape(policy, (batch_size, n_nodes))

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [H(B, N, D), indice(B, F), mask(B, N)]
        """
        H = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        # B, 1, D
        query = self.source_generator([H, indice])

        # B, 1, D
        policy = self._calc_policy(query, H, mask)
        return policy


class WouterDecoder(tf.keras.models.Model):

    def __init__(self, n_heads, d_key, th_range):
        super().__init__()
        self.d_key = d_key
        self.th_range = th_range
        self.source_generator = WouterSourceGenerator()
        self.mha = tf.keras.layers.MultiHeadAttention(n_heads, d_key)

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        initializer = tf.keras.initializers.GlorotNormal()
        self.wq = self.add_weight(name="wq", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)
        super().build(input_shape)

    def _calc_policy(self, query, H, mask):
        INFINITE = 1e+9
        Q = tf.matmul(query, self.wq)
        K = tf.matmul(H, self.wk)
        scale = tf.sqrt(float(self.d_key))
        # B, 1, N
        QK = tf.matmul(Q, K, transpose_b=True) / scale
        # B, 1, N
        mask = tf.cast(mask, tf.float32)
        mask = (1 - mask) * (-INFINITE)
        policy = tf.keras.activations.softmax(
            self.th_range * tf.keras.activations.tanh(QK) + mask)
        # now policy is tensor of shape(B, 1, N) which must be turned into tensor of
        # shape(B, N)
        return tf.squeeze(policy, axis=1)

    def call(self, inputs: List[tf.Tensor]):
        """

        Args:
            inputs (List[tf.Tensor]): [H(B, N, D), indice(B, F), mask(B, N)]
        """
        H = inputs[0]
        indice = inputs[1]
        mask = inputs[2]

        # B, 1, D
        query = self.source_generator([H, indice])

        # B, 1, N
        mask = tf.expand_dims(mask, axis=1)
        # B, 1, D
        query = self.mha(query, H, H, mask)

        # B, 1, D
        policy = self._calc_policy(query, H, mask)
        return policy
