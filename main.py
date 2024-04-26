import pandas as pd
import tensorflow as tf
from Config import Config
from CustomSchedule import CustomSchedule
from CustomTokenizer import CustomTokenizer
from Dataset import Dataset
from DataTrain import DataSet
from HierarchicalTransformerEncoderModel import HierarchicalTransformerEncoderModel
from Tokenizer import CreateVocab, Tokenizer
import tempfile
import six

config = Config().save_config()


# Read data.
data = pd.read_csv("data/vi_processed.csv")["correct_text"]
create_vocab = CreateVocab(special_token=['[PAD]', '[MASK]','[UNK]'])
word_vocab = create_vocab.create_word_vocab(file_text=data, vocab_size=config.VOCAB_SIZE)
char_vocab = create_vocab.create_character_vocab(file_text=data)
with tempfile.NamedTemporaryFile(delete=False) as word_vocab_writer:
    if six.PY2:
        word_vocab_writer.write("".join([x + "\n" for x in word_vocab]))
    else:
        word_vocab_writer.write("".join(
            [x + "\n" for x in word_vocab]).encode("utf-8"))

        word_vocab_file = word_vocab_writer.name

with tempfile.NamedTemporaryFile(delete=False) as char_vocab_writer:
    if six.PY2:
        char_vocab_writer.write("".join([x + "\n" for x in char_vocab]))
    else:
        char_vocab_writer.write("".join(
            [x + "\n" for x in char_vocab]).encode("utf-8"))

        char_vocab_file = char_vocab_writer.name


word_level_tokenizer = Tokenizer(vocab_file=word_vocab_file)
char_level_tokenizer = Tokenizer(vocab_file=char_vocab_file, char_level=True)


input_sentences = data[:config.NUM_OF_INPUTS]
dataset = DataSet(word_tokenizer=word_level_tokenizer, char_tokenizer=char_level_tokenizer,max_seq_length=config.MAX_SENTENCE_LENGTH, max_word_length=config.MAX_WORD_LENGTH)

(target_sequences, input_word_level_sequences, input_character_level_sequences) = dataset.create_data_train(input_sentences)

# Build and train model.
model = HierarchicalTransformerEncoderModel(num_character_level_layers=config.NUM_CHARACTER_LEVEL_LAYERS,
                                            num_word_level_layers=config.NUM_WORD_LEVEL_LAYERS,
                                            character_level_d_model=config.CHARACTER_LEVEL_D_MODEL,
                                            word_level_d_model=config.WORD_LEVEL_D_MODEL,
                                            num_heads=config.NUM_HEADS, dff=config.DFF,
                                            max_word_length=config.MAX_WORD_LENGTH,
                                            max_sentence_length=config.MAX_SENTENCE_LENGTH,
                                            vocab_size=config.VOCAB_SIZE,
                                            character_vocab_size=config.CHARACTER_VOCAB_SIZE)

learning_rate = CustomSchedule(config.WORD_LEVEL_D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
train_size = int(config.NUM_OF_INPUTS * 0.8)
model.fit(
    [input_word_level_sequences[:train_size], input_character_level_sequences[:train_size]],
    target_sequences[:train_size], epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_data=([input_word_level_sequences[train_size:], input_character_level_sequences[train_size:]],
                     target_sequences[train_size:]))

model.save('model/model.keras')
