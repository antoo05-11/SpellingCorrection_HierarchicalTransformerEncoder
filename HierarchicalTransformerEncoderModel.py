import tensorflow as tf

from Encoder import Encoder
from PositionalEmbedding import PositionalEmbedding

from tensorflow import keras


@keras.utils.register_keras_serializable()
class HierarchicalTransformerEncoderModel(tf.keras.models.Model):
    def __init__(self, *, 
                 num_character_level_layers, 
                 num_word_level_layers,
                 character_level_d_model, 
                 word_level_d_model, 
                 num_heads, 
                 dff,
                 max_word_length, 
                 max_sentence_length,
                 vocab_size, 
                 character_vocab_size, 
                 dropout_rate=0.1, **kwargs):
        super(HierarchicalTransformerEncoderModel, self).__init__(**kwargs)

        self.num_character_level_layers = num_character_level_layers
        self.num_word_level_layers = num_word_level_layers
        self.character_level_d_model = character_level_d_model
        self.word_level_d_model = word_level_d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_word_length = max_word_length
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.character_vocab_size = character_vocab_size
        self.dropout_rate = dropout_rate

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

        self.correction_layer = tf.keras.layers.Dense(vocab_size, activation='softmax', name='correction_layer', )

    def call(self, inputs):
        # print("Start")
        word_level_inputs, character_level_inputs = inputs

        word_embedding_outputs = self.word_pos_embedding(word_level_inputs)  # (batch_size, max_len, word_level_d_model)
        word_level_mask = self.word_pos_embedding.compute_mask(word_level_inputs)
        word_level_mask = word_level_mask[:, tf.newaxis, :]

        character_level_encoder_outputs = tf.map_fn(
            lambda sentence: self.character_level_encoder(
                (self.character_pos_embedding(sentence),
                 self.character_pos_embedding.compute_mask(sentence)[:, tf.newaxis, :])),
            character_level_inputs,
            dtype=tf.float32
        )
        # (batch_size, max_sen_len, max_word_length, character_level_d_model)

        character_level_encoder_outputs = tf.map_fn(
            lambda sentence: tf.map_fn(
                lambda word: tf.squeeze(self.linear_layer(self.flatten_layer(tf.expand_dims(word, axis=0))), axis=0),
                sentence,
                dtype=tf.float32
            ),
            character_level_encoder_outputs,
            dtype=tf.float32)
        # (batch_size, max_sen_len, word_level_d_model)
        # print("Shape của character_level_encoder_outputs:", character_level_encoder_outputs.shape)

        concat_output = self.combined_layer([word_embedding_outputs, character_level_encoder_outputs])
        # (batch_size, max_sen_len, word_level_d_model * 2)
        # print("Shape của concat_output:", concat_output.shape)

        word_level_output = self.word_level_encoder((concat_output, word_level_mask))
        # (batch_size, max_sen_len, word_level_d_model + character_level_d_model)
        # print("Shape của word_level_output:", word_level_output.shape)

        correction_output = self.correction_layer(word_level_output)  # (batch_size, max_sen_len, vocab_size)
        # print("Shape của correction_output:", correction_output.shape)

        # detection_output = tf.squeeze(self.detection_layer(word_level_output), axis=-1)  # (batch_size, max_sen_len)
        # print("Shape của detection_output:", detection_output.shape)
        # print("End")
        return correction_output

    def get_config(self):
        config = {
            'num_character_level_layers': self.num_character_level_layers,
            'num_word_level_layers': self.num_word_level_layers,
            'character_level_d_model': self.character_level_d_model,
            'word_level_d_model': self.word_level_d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'max_word_length': self.max_word_length,
            'max_sentence_length': self.max_sentence_length,
            'vocab_size': self.vocab_size,
            'character_vocab_size': self.character_vocab_size,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(HierarchicalTransformerEncoderModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.function
def training_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        correction_output = model(x)
        loss1 = correction_loss(y, correction_output)
        # loss2 = detection_loss(y[1], detection_output)
        total_loss = loss1

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss


def correction_loss(true_outputs, pred_outputs):
    softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(true_outputs, pred_outputs)
    return softmax_loss


def detection_loss(true_detection_infos, pred_detection_infos):
    sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(true_detection_infos, pred_detection_infos)
    return sigmoid_loss
