import json

from utils import write_dict_data_to_file


class Config:
    def __init__(self):
        self.BATCH_SIZE = 200
        self.EPOCHS = 30
        self.NUM_OF_INPUTS = 20000

        self.NUM_CHARACTER_LEVEL_LAYERS = 4
        self.NUM_WORD_LEVEL_LAYERS = 12

        self.CHARACTER_LEVEL_D_MODEL = 64
        self.WORD_LEVEL_D_MODEL = 768

        self.NUM_HEADS = 6
        self.DFF = 256

        self.MAX_WORD_LENGTH = 8
        self.MAX_SENTENCE_LENGTH = 64

        self.VOCAB_SIZE = 10000
        self.CHARACTER_VOCAB_SIZE = 500

    def save_config(self):
        write_dict_data_to_file("model/config.json", {
            "BATCH_SIZE": self.BATCH_SIZE,
            "EPOCHS": self.EPOCHS,
            "NUM_OF_INPUTS": self.NUM_OF_INPUTS,

            "NUM_CHARACTER_LEVEL_LAYERS": self.NUM_CHARACTER_LEVEL_LAYERS,
            "NUM_WORD_LEVEL_LAYERS": self.NUM_WORD_LEVEL_LAYERS,

            "WORD_LEVEL_D_MODEL": self.WORD_LEVEL_D_MODEL,
            "CHARACTER_LEVEL_D_MODEL": self.CHARACTER_LEVEL_D_MODEL,

            "NUM_HEADS": self.NUM_HEADS,
            "DFF": self.DFF,

            "MAX_WORD_LENGTH": self.MAX_WORD_LENGTH,
            "MAX_SENTENCE_LENGTH": self.MAX_SENTENCE_LENGTH,

            "VOCAB_SIZE": self.VOCAB_SIZE,
            "CHARACTER_VOCAB_SIZE": self.CHARACTER_VOCAB_SIZE,
        }, indent=4)
        return self

    @staticmethod
    def load_config():
        instance = Config()
        config = json.load(open('model/config.json'))

        instance.BATCH_SIZE = config["BATCH_SIZE"]
        instance.EPOCHS = config["EPOCHS"]

        instance.NUM_OF_INPUTS = config["NUM_OF_INPUTS"]
        instance.NUM_CHARACTER_LEVEL_LAYERS = config["NUM_CHARACTER_LEVEL_LAYERS"]
        instance.NUM_WORD_LEVEL_LAYERS = config["NUM_WORD_LEVEL_LAYERS"]

        instance.WORD_LEVEL_D_MODEL = config["WORD_LEVEL_D_MODEL"]
        instance.CHARACTER_LEVEL_D_MODEL = config["CHARACTER_LEVEL_D_MODEL"]

        instance.NUM_HEADS = config["NUM_HEADS"]
        instance.DFF = config["DFF"]

        instance.MAX_WORD_LENGTH = config["MAX_WORD_LENGTH"]
        instance.MAX_SENTENCE_LENGTH = config["MAX_SENTENCE_LENGTH"]

        instance.VOCAB_SIZE = config["VOCAB_SIZE"]
        instance.CHARACTER_VOCAB_SIZE = config["CHARACTER_VOCAB_SIZE"]

        return instance