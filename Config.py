class Config:
    def __init__(self):
        self.BATCH_SIZE = 50
        self.EPOCHS = 30
        self.NUM_OF_INPUTS = 1000

        self.NUM_CHARACTER_LEVEL_LAYERS = 4
        self.NUM_WORD_LEVEL_LAYERS = 12

        self.CHARACTER_LEVEL_D_MODEL = 64
        self.WORD_LEVEL_D_MODEL = 512

        self.NUM_HEADS = 3
        self.DFF = 256

        self.MAX_WORD_LENGTH = 16
        self.MAX_SENTENCE_LENGTH = 64

        self.VOCAB_SIZE = 5000
        self.CHARACTER_VOCAB_SIZE = 500