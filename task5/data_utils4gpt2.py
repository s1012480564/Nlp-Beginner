# -*- coding: utf-8 -*-
# file: data_utils4gpt2.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.


import torch
from torch.utils.data import Dataset


def data_collator(batch, args):
    tokens_dict = args.tokenizer.batch_encode_plus(batch, padding=True, is_split_into_words=True,
                                                   max_length=args.max_seq_len, truncation=True, return_tensors='pt')
    X = tokens_dict['input_ids']
    attn_mask = tokens_dict['attention_mask']
    lens = torch.sum(attn_mask != 0, dim=-1) - 2
    return X, attn_mask, lens


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
                self.data.append(tokenizer.encode(poem, add_special_tokens=False))
                poem = ''
        if poem:
            self.data.append(tokenizer.encode(poem, add_special_tokens=False))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
