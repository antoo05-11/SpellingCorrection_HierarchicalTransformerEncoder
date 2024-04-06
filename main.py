import json

from tensorflow.keras.preprocessing.text import Tokenizer

from Config import Config

# Read data.
json_sentences = []

with open('data/VSEC.jsonl', 'r') as file:
    for line in file:
        json_obj = json.loads(line[:-1])
        json_sentences.append(json_obj)

input_sentences = []
output_sentences = []
correct_infos = []

for json_sentence in json_sentences:
    input_sentence = []
    output_sentence = []
    correct_info = []
    for word in json_sentence['annotations']:
        correct_info.append(word['is_correct'])

        current_word = word['current_syllable'].lower()
        input_sentence.append(current_word)

        if word['is_correct'] is True:
            output_sentence.append(current_word)
        else:
            output_sentence.append(word['alternative_syllables'][0].lower())
            # if len(word['alternative_syllables']) > 1: print(word)
    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    correct_infos.append(correct_info)
# Main
config = Config()

word_level_tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token='<UNK>', lower=True)
word_unk_level_tokenizer = Tokenizer(oov_token='<UNK>', lower=True)
character_level_tokenizer = Tokenizer(num_words=config.CHARACTER_VOCAB_SIZE, lower=True, char_level=True)

word_level_tokenizer.fit_on_texts(output_sentences)
word_unk_level_tokenizer.fit_on_texts(input_sentences)
character_level_tokenizer.fit_on_texts(input_sentences + output_sentences)