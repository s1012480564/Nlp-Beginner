# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import numpy as np
import os
import pickle
from torch.utils.data import Dataset


def pad_and_truncate(tokens, max_len, dtype='int64', value=0):
    x = (np.ones(max_len) * value).astype(dtype)
    trunc = tokens[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 1  # 隐含第0个[PAD]。最后一个[UNK]不算在size中

    def fit(self, sentence):
        words = sentence.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def transform(self, sentence):
        words = sentence.split()
        unk_idx = self.vocab_size
        tokens = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        if len(tokens) == 0:
            tokens = [0]
        return pad_and_truncate(tokens, self.max_seq_len)


def build_tokenizer(path, max_seq_len, dat_fname, dat_fname4labels):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
        print('loading tokenizer4labels:', dat_fname4labels)
        tokenizer4labels = pickle.load(open(dat_fname4labels, 'rb'))
    else:
        document = ''
        document4labels = ''
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = line.split()
            if line[0] == '-DOCSTART-':
                continue
            document += line[0] + ' '
            document4labels += line[3] + ' '
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit(document)
        tokenizer4labels = Tokenizer(max_seq_len)
        tokenizer4labels.fit(document4labels)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
        pickle.dump(tokenizer4labels, open(dat_fname4labels, 'wb'))
    return tokenizer, tokenizer4labels


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # [PAD] and [UNK] are all-zeros
        fname = 'pre_train/glove.twitter.27B/glove.twitter.27B.' + str(
            embed_dim) + 'd.txt' if embed_dim != 300 else 'pre_train/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:  # words not found ([UNK]) in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


class NERDataset(Dataset):
    def __init__(self, path, tokenizer, tokenizer4labels):
        super().__init__()
        self.data = []
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        words = []
        labels = []
        for line in lines:
            line = line.strip()
            if line:
                line = line.split()
                if line[0] == '-DOCSTART-':
                    continue
                words.append(line[0])
                labels.append(line[3])
            elif words:  # 遇到空白行，DOC结束，但是要注意DOCSTART下一行的空白以及连续空白行导致空白句子的情况
                sentence_tokens = tokenizer.transform(' '.join(words))
                labels_tokens = tokenizer4labels.transform(' '.join(labels))
                words = []
                labels = []
                self.data.append((sentence_tokens, labels_tokens))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
