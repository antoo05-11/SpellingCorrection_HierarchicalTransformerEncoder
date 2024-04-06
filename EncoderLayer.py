from GlobalSelfAttention import GlobalSelfAttention
from FeedForward import FeedForward
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, inputs):
        x = self.self_attention(inputs)
        x = self.ffn(x)
        return x
