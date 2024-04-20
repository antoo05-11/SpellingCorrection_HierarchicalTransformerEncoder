import pandas as pd
import tensorflow as tf
from Config import Config
from CustomSchedule import CustomSchedule
from CustomTokenizer import CustomTokenizer
from Dataset import Dataset
from HierarchicalTransformerEncoderModel import HierarchicalTransformerEncoderModel

config = Config().save_config()

# Read data.
data = pd.read_csv('data/vi_processed.csv')
input_sentences = []
for index, row in data.iterrows():
    if len(input_sentences) == config.NUM_OF_INPUTS: break
    input_sentences.append(row.correct_text)
correct_texts = input_sentences[:config.NUM_OF_INPUTS]

# Preprocessing and build dataset.
word_level_tokenizer = CustomTokenizer(word_vocab_size=config.VOCAB_SIZE, max_word_len=config.MAX_WORD_LENGTH,
                                       max_sentence_len=config.MAX_SENTENCE_LENGTH)
word_level_tokenizer.fit_on_texts(input_sentences[:20000])

dataset = Dataset(input_sentences, word_level_tokenizer, config)
data = dataset.build_dataset()

(input_word_level_sequences, input_character_level_sequences), target_sequences = data

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
