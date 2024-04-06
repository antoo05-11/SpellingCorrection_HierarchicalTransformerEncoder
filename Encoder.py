import tensorflow as tf

from EncoderLayer import EncoderLayer


def create_padding_mask(input_tokens, actual_lengths):
    batch_size, sentence_len = tf.shape(input_tokens)[0], tf.shape(input_tokens)[1]

    mask = tf.sequence_mask(actual_lengths, maxlen=sentence_len, dtype=tf.float32)

    mask = tf.expand_dims(mask, axis=1)

    return 1 - mask


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        # self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        # `x` is token-IDs shape: (batch, seq_len)
        # x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Add dropout.
        x, lengths = inputs
        mask = create_padding_mask(x, lengths)
        print(mask.shape)
        for i in range(self.num_layers):
            inputs = self.enc_layers[i]((x, mask))

        return inputs  # Shape `(batch_size, seq_len, d_model)`.
