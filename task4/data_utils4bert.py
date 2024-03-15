# -*- coding: utf-8 -*-
# file: data_utils4bert.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch.utils.data import Dataset


def build_label2idx(path):
    label2idx = {'[PAD]': 0}
    target_size = 1
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            line = line.split()
            if line[0] == '-DOCSTART-':
                continue
            label = line[3]
            if label not in label2idx:
                label2idx[label] = target_size
                target_size += 1
    return label2idx, target_size


def data_collator(batch, args):
    X_list = [elem[0] for elem in batch]
    y_list = [elem[1] for elem in batch]
    tokens_dict = args.tokenizer.batch_encode_plus(X_list, padding=True, is_split_into_words=True,
                                                   max_length=args.max_seq_len, truncation=True, return_tensors='pt')
    X = tokens_dict['input_ids']
    attn_mask = tokens_dict['attention_mask']
    max_len = X.shape[1]
    y = args.tokenizer.batch_encode_plus(y_list, padding='max_length', max_length=max_len,
                                         is_split_into_words=True, truncation=True, return_tensors='pt')['input_ids']
    y = y[:, 1:]  # remove [CLS]
    lens = torch.sum(attn_mask != 0, dim=-1) - 2
    mask = attn_mask.clone()[:, 1:]  # remove [CLS]
    mask[range(len(mask)), lens] = 0  # remove [SEP]
    y = y * mask  # remove [SEP]
    return X, y, attn_mask, mask.to(torch.bool), lens


class NERDataset(Dataset):
    def __init__(self, path, tokenizer, label2idx):
        self.data = []
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        sentence_tokens = []
        labels_tokens = []
        for line in lines:
            line = line.strip()
            if line:
                line = line.split()
                if line[0] == '-DOCSTART-':
                    continue
                # BERT问题很大，单词会被拆成 ## 前后两部分，长度变长了，位置对应的也不对了，特别离谱
                # 对于单词可能被拆分，这里我直接保留第一个词，当然，靠谱的做法是词向量取平均值。这要写起来就乱套了，超级麻烦，还是算了
                sentence_tokens.append(tokenizer.encode(line[0], add_special_tokens=False)[0])
                labels_tokens.append(label2idx[line[3]])
            elif sentence_tokens:  # 遇到空白行，DOC结束，但是要注意DOCSTART下一行的空白以及连续空白行导致空白句子的情况
                sentence_tokens = []
                labels_tokens = []
                self.data.append((sentence_tokens, labels_tokens))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
