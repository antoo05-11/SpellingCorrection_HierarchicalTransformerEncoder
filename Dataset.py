import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


class Dataset:
    def __init__(self, sentences, tokenizer, config):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.config = config

    def build_dataset(self):
        (word_level_sequences, character_level_sequences) = self.tokenizer.texts_to_sequences(self.sentences)

        target_word_level_sequences = pad_sequences(word_level_sequences.copy(), maxlen=self.config.MAX_SENTENCE_LENGTH,
                                                    padding='post', truncating='post')

        tmp_word_level_sequences = []
        for sequence in word_level_sequences:
            sequence = self.random_word(sequence)
            tmp_word_level_sequences.append(sequence)

        # Get word_level sequences
        word_level_sequences = tmp_word_level_sequences
        word_level_sequences = pad_sequences(word_level_sequences, maxlen=self.config.MAX_SENTENCE_LENGTH,
                                             padding='post', truncating='post')

        # Get character_level mask matrix.
        tmp_character_level_sequences = []
        for character_level_sequence in character_level_sequences:
            sentence_tmp = []
            # With each sentence
            for word in character_level_sequence:
                # With each word
                character_level_sequence = self.random_word(word, word_level=False)
                sentence_tmp.append(character_level_sequence)

            while len(sentence_tmp) < self.config.MAX_SENTENCE_LENGTH:
                sentence_tmp.append([])
            while len(sentence_tmp) > self.config.MAX_SENTENCE_LENGTH:
                del sentence_tmp[-1]
            tmp_character_level_sequences.append(
                pad_sequences(sentence_tmp, maxlen=self.config.MAX_WORD_LENGTH,
                              padding='post', truncating='post'))

        character_level_sequences = tf.convert_to_tensor(tmp_character_level_sequences)

        return (word_level_sequences, character_level_sequences), target_word_level_sequences

    def random_word(self, sequence, word_level=True):
        output = []

        # 15% of the tokens would be replaced
        for token in sequence:
            prob = random.random()

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(0)

                # 10% chance change token to random token
                elif prob < 0.9:
                    if word_level:
                        random_token = random.randrange(len(self.tokenizer.word_index) + 1)
                        if random_token > self.tokenizer.word_vocab_size:
                            random_token = 0
                    else:
                        random_token = random.randrange(len(self.tokenizer.character_index) + 1)

                    output.append(random_token)

                # 10% chance change token to current token
                else:
                    output.append(token)

            else:
                output.append(token)

        return output

    def build_test_data(self, sentences):
        word_level_sequences, character_level_sequences = self.tokenizer.texts_to_sequences(sentences)

        word_level_sequences = pad_sequences(word_level_sequences, maxlen=self.config.MAX_SENTENCE_LENGTH,
                                             padding='post', truncating='post')

        tmp_character_level_sequences = []
        for character_level_sequence in character_level_sequences:
            while len(character_level_sequence) < self.config.MAX_SENTENCE_LENGTH:
                character_level_sequence.append([])
            while len(character_level_sequence) > self.config.MAX_SENTENCE_LENGTH:
                del character_level_sequence[-1]
            tmp_character_level_sequences.append(
                pad_sequences(character_level_sequence, maxlen=self.config.MAX_WORD_LENGTH,
                              padding='post', truncating='post'))

        character_level_sequences = tf.convert_to_tensor(tmp_character_level_sequences)
        return word_level_sequences, character_level_sequences
