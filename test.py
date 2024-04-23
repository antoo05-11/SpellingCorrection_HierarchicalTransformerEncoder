from Config import Config
from CustomTokenizer import CustomTokenizer
import tensorflow as tf
from HierarchicalTransformerEncoderModel import HierarchicalTransformerEncoderModel
from BaseAttention import BaseAttention
from GlobalSelfAttention import GlobalSelfAttention
from CustomSchedule import CustomSchedule
from CustomTokenizer import CustomTokenizer
from Dataset import Dataset
from Encoder import Encoder
from EncoderLayer import EncoderLayer
from FeedForward import FeedForward
from PositionalEmbedding import PositionalEmbedding, positional_encoding

tokenizer = CustomTokenizer.load_tokenizer()
loaded_model = tf.keras.models.load_model('model/model.keras')
config = Config.load_config()
dataset = Dataset(None, tokenizer, config)


def print_outputs(outputs):
    for sentence in outputs:
        out = ''
        for word in sentence:
            max_index = tf.argmax(word, axis=0).numpy()
            word_str = tokenizer.index_word.get(str(max_index))
            if word_str is not None:
                out += word_str + ' '
            else:
                out += '<UNK> '
        print(out)


test_sentences = ['Nếu làm được như vậy thì chắc chắn xẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng củb phụ huynh và ai không có tiền thì hhông cần đóng.']
test_inputs = dataset.build_test_data(test_sentences)
test_outputs = loaded_model.predict(test_inputs)
print_outputs(test_outputs)