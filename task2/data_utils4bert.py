# -*- coding: utf-8 -*-
# file: data_utils4bert.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import pandas as pd
from torch.utils.data import Dataset


def data_collator(batch, args):
    X_list = [elem[0] for elem in batch]
    y_list = [elem[1] for elem in batch]
    tokens_dict = args.tokenizer.batch_encode_plus(X_list, padding=True, is_split_into_words=True,
                                                   max_length=args.max_seq_len, truncation=True, return_tensors='pt')
    X = tokens_dict['input_ids']
    attn_mask = tokens_dict['attention_mask']
    y = torch.tensor(y_list)
    return X, y, attn_mask


class TextDataset(Dataset):
    def __init__(self, path, tokenizer):
        df = pd.read_csv(path)
        tokens = []
        for phrase in df['Phrase']:
            phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
            if len(phrase_tokens) == 0:
                phrase_tokens.append(tokenizer.unk_token_id)
            tokens.append(phrase_tokens)
        self.data = list(zip(tokens, df['Sentiment'].values))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
