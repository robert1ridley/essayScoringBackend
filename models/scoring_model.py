import re
import os
import json
import jieba
import xlrd
import numpy as np
import tensorflow as tf
from custom_layers.attention import Attention
from custom_layers.zeromasking import ZeroMaskedEntries


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


class ScoringModel(object):
    def __init__(self, essay_text, vocab_loc, model_loc, wordlists_paths, idiom_path):
        self.essay_text = essay_text
        self.vocab_loc = vocab_loc
        self.model_loc = model_loc
        self.wordlists_paths = wordlists_paths
        self.idiom_path = idiom_path
        self.idioms_clean = None
        self.hsk_word_dict = None
        self.sentences = None
        self.words = None
        self.vocab = None
        self.idiom_prop = None
        self.high_prop = None
        self.model = None
        self.padded_text = None

    def load_vocab(self):
        with open(self.vocab_loc) as f:
            self.vocab = json.load(f)

    def load_idioms(self):
        idioms = open(self.idiom_path, 'r').readlines()
        self.idioms_clean = [id.strip() for id in idioms]

    def load_hsk_wordlists(self):
        hsk_word_dict = {}
        for path_ in self.wordlists_paths:
            f, e = os.path.splitext(path_)
            filename_split = f.split('/')
            filename = filename_split[-1]
            wb = xlrd.open_workbook(path_)
            sheet = wb.sheet_by_index(0)
            words = sheet.col_values(2, 2)
            hsk_word_dict[filename] = words
        self.hsk_word_dict = hsk_word_dict

    def get_idiom_prop(self):
        idiom_count = 0
        for idiom in self.idioms_clean:
            if idiom in self.essay_text:
                idiom_count += 1
        words = []
        for sent in self.sentences:
            for word in sent:
                words.append(word)
        self.words = words
        self.idiom_prop = idiom_count / len(words)

    def get_high_vocab_prop(self):
        word_level_counts = {'unk': 0}
        word_dict_names = self.get_word_dict_names()
        for word in self.words:
            in_dict = False
            for word_dict_name in word_dict_names:
                if word in self.hsk_word_dict[word_dict_name]:
                    in_dict = True
                    new_level = self.get_level_matches(word_dict_name)
                    if new_level in word_level_counts.keys():
                        word_level_counts[new_level] += 1
                    else:
                        word_level_counts[new_level] = 1
                    break
            if in_dict == False:
                word_level_counts['unk'] += 1
        for k in word_level_counts.keys():
            word_level_counts[k] = word_level_counts[k] / len(self.words)
        try:
            self.high_prop = word_level_counts[3]
        except KeyError:
            self.high_prop = 0

    def get_vocab_score(self):
        idiom_scores = {
            1: (0, 0.012),
            2: (0.012, 0.024),
            3: (0.024, 0.036),
            4: (0.036, 0.06),
            5: (0.06, 1)
        }
        high_voc_scores = {
            1: (0, 0.024),
            2: (0.024, 0.048),
            3: (0.048, 0.072),
            4: (0.072, 0.096),
            5: (0.096, 1)
        }
        idiom_score = 0
        high_score = 0
        for k in idiom_scores:
            if self.idiom_prop >= idiom_scores[k][0] and self.idiom_prop < idiom_scores[k][1]:
                idiom_score = k
        for k in high_voc_scores:
            if self.high_prop >= high_voc_scores[k][0] and self.high_prop < high_voc_scores[k][1]:
                high_score = k
        print(self.high_prop)
        print(self.idiom_prop)
        return idiom_score + high_score

    def get_readability_score(self):
        '''
        Using The Flesch Reading Ease Readability Formula:
        RE = 206.835 – (1.015 x ASL) – (84.6 x ASW)
        ASL = Average Sentence Length
        ASW = Average Syllables per Word
        :return: int (0-10)
        '''
        sentence_lengths = [len(sentence) for sentence in self.sentences]
        word_lengths = [len(word) for sentence in self.sentences for word in sentence]
        ASL = np.mean(sentence_lengths)
        ASW = np.mean(word_lengths)
        RE = 206.835 - (1.015*ASL) - (84.6*ASW)
        if RE < 0:
            RE = 0
        elif RE > 100:
            RE = 100
        readability_score = int(np.ceil((100 - RE)/10))
        return readability_score

    def load_model(self):
        layers_dict = {'ZeroMaskedEntries': ZeroMaskedEntries, 'Attention': Attention}
        self.model = tf.keras.models.load_model(self.model_loc, custom_objects=layers_dict)

    def process_essay_text(self, max_sent=64, max_word=50): # max vals hardcoded
        self.sentences = self.cut_sent(self.essay_text)
        sentences_ids = self.convert_text_to_ids(self.sentences)
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

    @staticmethod
    def get_level_matches(in_level):
        level_dict = {
            'HSK_Level_1': 1,
            'HSK_Level_2': 1,
            'HSK_Level_3': 2,
            'HSK_Level_4': 2,
            'HSK_Level_5': 3,
            'HSK_Level_6': 3
        }
        return level_dict[in_level]

    @staticmethod
    def get_word_dict_names():
        return [
            'HSK_Level_1',
            'HSK_Level_2',
            'HSK_Level_3',
            'HSK_Level_4',
            'HSK_Level_5',
            'HSK_Level_6'
        ]