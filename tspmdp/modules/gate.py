import tensorflow as tf


class GRUGate(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        D = input_shape[0][-1]
        self.w_r = tf.keras.layers.Dense(D, use_bias=False)
        self.w_z = tf.keras.layers.Dense(D, use_bias=False)
        self.w_g = tf.keras.layers.Dense(D, use_bias=False)
        self.u_r = tf.keras.layers.Dense(D, use_bias=False)
        self.u_z = tf.keras.layers.Dense(D, use_bias=False)
        self.u_g = tf.keras.layers.Dense(D, use_bias=False)
        #
        self.b_g = tf.Variable(tf.ones((1, 1, D), dtype=tf.float32) * 2)

    def call(self, inputs):
        # B, N, D
        x = inputs[0]
        # B, N, D
        y = inputs[1]
        # B, N, D
        r = tf.keras.activations.sigmoid(self.w_r(y) + self.u_r(x))
        # B, N, D
        z = tf.keras.activations.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g)
        # B, N, D
        h = tf.keras.activations.tanh(self.w_g(y) + self.u_g(r * x))
        return (1-z) * x + z * h
