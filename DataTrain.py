import collections
import tensorflow as tf
import numpy as np
import random
from Tokenizer import Tokenizer
class DataSet(object):
  def __init__(self, word_tokenizer:Tokenizer, char_tokenizer:Tokenizer, max_seq_length=64, max_word_length = 10, padding_token = "[PAD]"):
      self.word_tokenizer = word_tokenizer
      self.char_tokenizer = char_tokenizer
      self.max_seq_length = max_seq_length
      self.max_word_length = max_word_length
      self.padding_token = padding_token
  def create_data_train(self, list_seq):
    target_seq = [self.word_tokenizer.convert_tokens_to_ids(padding(self.word_tokenizer.tokenize(seq),self.max_seq_length,self.padding_token)) for seq in list_seq]
    word_input = tf.convert_to_tensor(self._create_word_input(list_seq=list_seq))
    char_input = self._create_char_input(list_seq=list_seq)
    target_seq = tf.convert_to_tensor(target_seq)
    return (target_seq, word_input, char_input)
    # if list_seq: print(len(list_seq))
  def _create_word_input(self, list_seq):
    origin_input = [self.word_tokenizer.tokenize(seq) for seq in list_seq]
    word_input = []
    for seq in origin_input:
      seq = create_masked_lm_predictions(tokens=seq, vocab_words=self.word_tokenizer.inv_vocab)
      seq = padding(seq, max_length=self.max_seq_length, padding_token=self.padding_token)
      seq = self.word_tokenizer.convert_tokens_to_ids(seq)
      word_input.append(seq)
    return word_input
  def _create_char_input(self, list_seq):
    char_input = []
    origin_input = [self.char_tokenizer.tokenize(seq) for seq in list_seq]
    for seq in origin_input:
      seq_padding = padding(seq, self.max_seq_length, padding_token=['[PAD]'])
      tokens = []
      for word in seq_padding:
        word = self.char_tokenizer.convert_tokens_to_ids(padding(word, self.max_word_length, self.padding_token))
        tokens.append(word)
        # word = self.char_tokenizer.convert_tokens_to_ids(word)
      char_input.append(tokens)
    return char_input
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
def create_masked_lm_predictions(tokens, vocab_words, rng = random.Random(12345), masked_lm_prob = 0.15,
                                 max_predictions_per_seq = 20):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return output_tokens

def padding(input_ids, max_length, padding_token):
    len_sentence = len(input_ids)
    while len(input_ids) > max_length:
      del input_ids[-1]
    for i in range(max_length - len_sentence):
      input_ids.append(padding_token)
    return input_ids
