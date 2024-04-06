import tensorflow as tf

from Encoder import Encoder
from PositionalEmbedding import PositionalEmbedding


class HierarchicalTransformerEncoder(tf.keras.models.Model):
    def __init__(self, *, num_character_level_layers, num_word_level_layers,
                 character_level_d_model, word_level_d_model, num_heads, dff,
                 max_word_length, max_sentence_length,
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
                                               dropout_rate=dropout_rate,
                                               name='character_level_encoder')

        self.flatten_layer = tf.keras.layers.Flatten(input_shape=(max_word_length, character_level_d_model),
                                                     name='flatten_character_level_encoders')
        self.linear_layer = tf.keras.layers.Dense(units=word_level_d_model, activation=None,
                                                  name='linear_character_level_encoders')

        self.combined_layer = tf.keras.layers.Concatenate(axis=-1, name='combined_character_and_word_level_encoders')

        self.word_level_encoder = Encoder(num_layers=num_word_level_layers,
                                          d_model=(word_level_d_model * 2),
                                          num_heads=num_heads, dff=dff,
                                          vocab_size=vocab_size,
                                          dropout_rate=dropout_rate,
                                          name='word_level_encoder')

        self.correction_layer = tf.keras.layers.Dense(vocab_size, activation='softmax', name='correction_layer',)
        # self.detection_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        (word_level_inputs, sentences_lengths), (character_level_inputs, words_lengths) = inputs

        word_embedding_outputs = self.word_pos_embedding(word_level_inputs)
        # print("Shape của word_embedding_outputs:", word_embedding_outputs.shape)

        character_level_encoder_outputs = tf.map_fn(
            lambda sentence: self.character_level_encoder(
                (self.character_pos_embedding(sentence[0]), sentence[1])),
            (character_level_inputs, words_lengths),
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

        # print("Shape của character_level_encoder_outputs:", character_level_encoder_outputs.shape)

        concat_output = self.combined_layer([word_embedding_outputs, character_level_encoder_outputs])
        # print("Shape của concat_output:", concat_output.shape)

        word_level_output = self.word_level_encoder((concat_output, sentences_lengths))
        # print("Shape của word_level_output:", word_level_output.shape)

        correction_output = self.correction_layer(word_level_output)
        # print("Shape của correction_output:", correction_output.shape)

        # detection_output = self.detection_layer(word_level_output)
        # print("Shape của detection_output:", detection_output.shape)

        return correction_output


def custom_loss(y_true, y_pred):
    print('Find loss:')
    print("Shape y_true:", y_true.shape)
    print("Shape y_pred:", y_pred.shape)
    softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    print('loss = ', softmax_loss)
    # sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred[1])
    total_loss = softmax_loss
    return total_loss
