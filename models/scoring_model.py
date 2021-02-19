import re
import json
import jieba
import numpy as np
import tensorflow as tf
from custom_layers.attention import Attention
from custom_layers.zeromasking import ZeroMaskedEntries


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


class ScoringModel(object):
    def __init__(self, essay_text, vocab_loc, model_loc):
        self.essay_text = essay_text
        self.vocab_loc = vocab_loc
        self.model_loc = model_loc
        self.vocab = None
        self.model = None
        self.padded_text = None

    def load_vocab(self):
        with open(self.vocab_loc) as f:
            self.vocab = json.load(f)

    def load_model(self):
        layers_dict = {'ZeroMaskedEntries': ZeroMaskedEntries, 'Attention': Attention}
        self.model = tf.keras.models.load_model(self.model_loc, custom_objects=layers_dict)

    def process_essay_text(self, max_sent=64, max_word=50): # max vals hardcoded
        sentences = self.cut_sent(self.essay_text)
        sentences_ids = self.convert_text_to_ids(sentences)
        padded_text = self.pad_hierarchical_text_sequences(sentences_ids, max_sent, max_word)
        self.padded_text = padded_text.reshape((padded_text.shape[0], padded_text.shape[1] * padded_text.shape[2]))

    def predict_score(self):
        pred = self.model.predict(self.padded_text)
        print(pred)
        scaled_pred = self.rescale_scores(pred)
        print(scaled_pred)
        return int(scaled_pred[0, 0])

    def convert_text_to_ids(self, sentences):
        essay_texts = []
        sentences_list = []
        for sentence in sentences:
            sentence_ids = []
            for word in sentence:
                if self.is_number(word):
                    sentence_ids.append(self.vocab['<num>'])
                elif word in self.vocab.keys():
                    sentence_ids.append(self.vocab[word])
                else:
                    sentence_ids.append(self.vocab['<unk>'])
            sentences_list.append(sentence_ids)
        essay_texts.append(sentences_list)
        return essay_texts

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        para = para.rstrip()
        sents = para.split("\n")
        sent_tokens = []
        for sent in sents:
            shorten_sents_tokens = self.shorten_sentence(sent, max_sentlen=50)
            sent_tokens.extend(shorten_sents_tokens)
        return sent_tokens

    @staticmethod
    def shorten_sentence(sent, max_sentlen=50):
        new_tokens = []
        sent = sent.strip()
        words = jieba.cut(sent)
        tokens = [word for word in words]
        if len(tokens) > max_sentlen:
            split_keywords = ['因为', '但是', '所以', '不过', '因此', '此外', '可是', '从而', '不然', '无论如何', '由于']
            k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
            processed_tokens = []
            if not k_indexes:
                num = len(tokens) / max_sentlen
                num = int(round(num))
                k_indexes = [(i + 1) * max_sentlen for i in range(num)]

            processed_tokens.append(tokens[0:k_indexes[0]])
            len_k = len(k_indexes)
            for j in range(len_k - 1):
                processed_tokens.append(tokens[k_indexes[j]:k_indexes[j + 1]])
            processed_tokens.append(tokens[k_indexes[-1]:])

            for token in processed_tokens:
                if len(token) > max_sentlen:
                    num = len(token) / max_sentlen
                    num = int(np.ceil(num))
                    s_indexes = [(i + 1) * max_sentlen for i in range(num)]

                    len_s = len(s_indexes)
                    new_tokens.append(token[0:s_indexes[0]])
                    for j in range(len_s - 1):
                        new_tokens.append(token[s_indexes[j]:s_indexes[j + 1]])
                    new_tokens.append(token[s_indexes[-1]:])

                else:
                    new_tokens.append(token)
        else:
            return [tokens]

        return new_tokens

    @staticmethod
    def is_number(token):
        return bool(num_regex.match(token))

    @staticmethod
    def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
        X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

        for i in range(len(index_sequences)):
            sequence_ids = index_sequences[i]
            num = len(sequence_ids)

            for j in range(num):
                word_ids = sequence_ids[j]
                length = len(word_ids)
                for k in range(length):
                    wid = word_ids[k]
                    X[i, j, k] = wid
                X[i, j, length:] = 0

            X[i, num:, :] = 0
        return X

    @staticmethod
    def rescale_scores(scores_array, min_score=0, max_score=10):
        rescaled = scores_array * (max_score - min_score) + min_score
        return np.around(rescaled).astype(int)