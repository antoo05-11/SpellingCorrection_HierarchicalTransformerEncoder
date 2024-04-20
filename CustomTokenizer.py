import json

from utils import write_dict_data_to_file


class CustomTokenizer:

    def __init__(self, word_vocab_size=0, max_word_len=0, max_sentence_len=0):
        self.index_word = {1: '[UNK]'}
        self.word_count = {'[UNK]': 1}
        self.word_index = {'[UNK]': 1}
        self.word_vocab_size = word_vocab_size

        self.character_index = {'[UNK]': 1}
        self.index_character = {1: '[UNK]'}

        self.signs = [',', '.', '!', '\"', "\'", "?", ";", ")", "(", ':', "/", "+", "-", "=", "`", "~", "*", "^", "@",
                      "%", "&", "_"]

        self.max_word_len = max_word_len
        self.max_sentence_len = max_sentence_len

    def preprocess_texts(self, texts):
        texts_preprocessed = []
        for text in texts:
            for sign in self.signs:
                text = text.replace(sign, ' ')

            text = text.lower()
            texts_preprocessed.append(text)
        return texts_preprocessed

    def sort(self):
        self.word_count = sorted(list(self.word_count.items())[1:], key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(self.word_count):
            self.index_word[i + 2] = word
            self.word_index[word] = i + 2

        self.word_count = dict(self.word_count)

    def texts_to_sequences(self, texts):
        word_level_sequences = []
        character_level_sequences = []

        texts = self.preprocess_texts(texts)
        for text in texts:
            words = text.split(' ')
            words = [i for i in words if i != '']

            tokens = [self.word_index.get(word, 1)
                      if (self.word_index.get(word, 1) <= self.word_vocab_size) else 1
                      for word in words]
            word_level_sequences.append(tokens)

            sentence_tokens = []
            for word in words:
                tokens = [self.character_index.get(i, 1) for i in word]
                sentence_tokens.append(tokens)
            character_level_sequences.append(sentence_tokens)

        return word_level_sequences, character_level_sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            texts.append([self.index_word.get(token, 1) for token in sequence])
        return texts

    def fit_on_texts(self, texts):

        texts = self.preprocess_texts(texts)

        for text in texts:

            # Update word vocab.
            words = text.split(' ')
            for word in words:
                if word == '': continue
                self.word_count[word] = self.word_count.get(word, 0) + 1

            # Update character vocab
            for character in text:
                if character == ' ': continue
                if self.character_index.get(character, -1) == -1:
                    self.character_index[character] = len(self.character_index) + 1

                self.index_character[self.character_index[character]] = character

        self.sort()
        self.save_tokenizer()

    def save_tokenizer(self):
        write_dict_data_to_file("model/tokenizer/word_index.json", self.word_index)
        write_dict_data_to_file("model/tokenizer/index_word.json", self.index_word)
        write_dict_data_to_file("model/tokenizer/index_character.json", self.index_character)
        write_dict_data_to_file("model/tokenizer/character_index.json", self.character_index)
        write_dict_data_to_file("model/tokenizer/config.json", {
            'word_vocab_size': self.word_vocab_size,
            'max_word_len': self.max_word_len,
            'max_sentence_len': self.max_sentence_len,
            'signs': self.signs
        }, indent=4)

    @staticmethod
    def load_tokenizer():
        instance = CustomTokenizer()

        config = json.load(open('model/tokenizer/config.json'))
        instance.word_vocab_size = config['word_vocab_size']
        instance.max_word_len = config['max_word_len']
        instance.max_sentence_len = config['max_sentence_len']
        instance.signs = config['signs']

        instance.word_index = json.load(open('model/tokenizer/word_index.json'))
        instance.index_word = json.load(open('model/tokenizer/index_word.json'))
        instance.character_index = json.load(open('model/tokenizer/character_index.json'))
        instance.index_character = json.load(open('model/tokenizer/index_character.json'))

        return instance
