class Config:
    def __init__(self):
        self.BATCH_SIZE = 50
        self.EPOCHS = 10
        self.NUM_OF_INPUTS = 150

        self.NUM_CHARACTER_LEVEL_LAYERS = 3
        self.NUM_WORD_LEVEL_LAYERS = 6

        self.CHARACTER_LEVEL_D_MODEL = 32
        self.WORD_LEVEL_D_MODEL = 128

        self.NUM_HEADS = 3
        self.DFF = 128

        self.MAX_WORD_LENGTH = 10
        self.MAX_SENTENCE_LENGTH = 40

        self.VOCAB_SIZE = 10000
        self.CHARACTER_VOCAB_SIZE = 1000
