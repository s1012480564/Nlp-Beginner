# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import json
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
        self.word2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2word = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.vocab_size = 3  # [PAD]:0，[BOS]:1，[EOS]:2。最后一个[UNK]

    # 上面和下面都还得改一下：
    def fit(self, sentence):
        words = []
        for word in sentence:
            words.append(word)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        self.word2idx['[UNK]'] = self.vocab_size
        self.idx2word[self.vocab_size] = '[UNK]'
        self.vocab_size += 1

    def transform(self, sentence):
        words = []
        for word in sentence:
            words.append(word)
        words = ['[BOS]'] + words + ['[EOS]']
        unk_idx = self.vocab_size - 1
        tokens = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        return pad_and_truncate(tokens, self.max_seq_len)


def build_tokenizer(path, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        document = ''
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                document += line
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit(document)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


class PoetryDataset(Dataset):
    def __init__(self, path, tokenizer):
        super().__init__()
        self.data = []
        poem = ''
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                poem += line
            elif poem:
                self.data.append(tokenizer.transform(poem))
                poem = ''
        if poem:
            self.data.append(tokenizer.transform(poem))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
