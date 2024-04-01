import tensorflow as tf

from Encoder import Encoder
from PositionalEmbedding import PositionalEmbedding


class HierarchicalTransformerEncoder(tf.keras.models.Model):
    def __init__(self, *, num_character_level_layers, num_word_level_layers,
                 character_level_d_model, word_level_d_model, num_heads, dff,
                 vocab_size, character_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.word_pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                      d_model=word_level_d_model)
        self.character_pos_embedding = PositionalEmbedding(vocab_size=character_vocab_size,
                                                           d_model=character_level_d_model)

        self.character_level_encoder = Encoder(num_layers=num_character_level_layers,
                                               d_model=character_level_d_model,
                                               num_heads=num_heads, dff=dff,
                                               vocab_size=character_vocab_size,
                                               dropout_rate=dropout_rate)

        self.word_level_encoder = Encoder(num_layers=num_word_level_layers,
                                          d_model=(word_level_d_model * 2),
                                          num_heads=num_heads, dff=dff,
                                          vocab_size=vocab_size,
                                          dropout_rate=dropout_rate)
        self.linear_layer = tf.keras.layers.Dense(units=word_level_d_model, activation=None)
        self.flatten_layer = tf.keras.layers.Flatten(input_shape=(32, 100))

        self.combined_layer = tf.keras.layers.Concatenate(axis=-1)
        self.correction_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.detection_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        word_level_inputs, character_level_inputs = inputs
        print("Shape của word_level_inputs:", word_level_inputs.shape)
        word_embedding_outputs = self.word_pos_embedding(word_level_inputs)
        print("Shape của word_embedding_outputs:", word_embedding_outputs.shape)

        # test = self.character_pos_embedding(tf.stack(character_level_inputs[0], axis=0))
        # print(test.shape)
        # test = tf.stack(test, axis=0)
        # test = self.character_level_encoder(test)
        # print(test.shape)

        character_level_encoder_outputs = tf.map_fn(
            lambda sentence: self.character_level_encoder(self.character_pos_embedding(sentence)),
            character_level_inputs,
            dtype=tf.float32
        )

        character_level_encoder_outputs = tf.map_fn(
            lambda sentence: tf.map_fn(
                lambda word: tf.squeeze(self.linear_layer(self.flatten_layer(tf.expand_dims(word, axis=0))), axis=0),
                sentence,
                dtype=tf.float32
            ),
            character_level_encoder_outputs,
            dtype=tf.float32)

        print("Shape của character_level_encoder_outputs:", character_level_encoder_outputs.shape)

        concat_output = self.combined_layer([word_embedding_outputs, character_level_encoder_outputs])
        print("Shape của concat_output:", tf.shape(concat_output))

        word_level_output = self.word_level_encoder(concat_output)
        print("Shape của word_level_output:", word_level_output.shape)

        correction_output = self.correction_layer(word_level_output)
        print("Shape của correction_output:", correction_output.shape)

        detection_output = self.detection_layer(word_level_output)
        print("Shape của detection_output:", detection_output.shape)

        return correction_output, detection_output


def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    print('loss = ', softmax_loss)
    sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred[1])
    total_loss = softmax_loss + sigmoid_loss
    return total_loss
