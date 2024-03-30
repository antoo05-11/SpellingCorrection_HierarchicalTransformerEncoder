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
                                          d_model=word_level_d_model,
                                          num_heads=num_heads, dff=dff,
                                          vocab_size=vocab_size,
                                          dropout_rate=dropout_rate)

        self.combined_layer = tf.keras.layers.Concatenate(axis=-1)
        self.correction_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.detection_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, word_level_inputs, character_level_inputs):
        word_embedding_outputs = self.word_pos_embedding(word_level_inputs)

        character_level_inputs = [self.character_pos_embedding(character_level_input)
                                  for character_level_input in character_level_inputs]
        character_level_encoder_output = self.character_level_encoder(character_level_inputs)

        concat_output = self.combined_layer([word_embedding_outputs, character_level_encoder_output])

        word_level_output = self.word_level_encoder(concat_output)

        correction_output = self.correction_layer(word_level_output)
        detection_output = self.detection_layer(word_level_output)

        return correction_output, detection_output
