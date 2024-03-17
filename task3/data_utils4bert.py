# -*- coding: utf-8 -*-
# file: data_utils4bert.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import json
import torch
from torch.utils.data import Dataset


def data_collator(batch, args):
    X_list = [elem[0] for elem in batch]
    y_list = [elem[1] for elem in batch]
    tokens_dict = args.tokenizer.batch_encode_plus(X_list, padding=True, is_split_into_words=True,
                                                   max_length=args.max_seq_len, truncation=True, return_tensors='pt')
    X = tokens_dict['input_ids'].to(args.device)
    attn_mask = tokens_dict['attention_mask'].to(args.device)
    y = torch.tensor(y_list).to(args.device)

    max_len = X.shape[1]
    start_indexes = []
    end_indexes = []
    for i in range(1, max_len - 1):
        for j in range(i, max_len - 1):
            start_indexes.append(i)
            end_indexes.append(j)
    start_indexes = torch.tensor(start_indexes, device=args.device)
    end_indexes = torch.tensor(end_indexes, device=args.device)

    lens = torch.sum(attn_mask != 0, dim=-1)
    lens_mat = lens.reshape(-1, 1)
    middle_indexes_mat = torch.where(X == 2)[1][::2].reshape(-1, 1)
    start_indexes_ex = start_indexes.expand((X.shape[0], -1))
    end_indexes_ex = end_indexes.expand((X.shape[0], -1))
    span_masks_bool = (start_indexes_ex <= lens_mat - 2) & (end_indexes_ex <= lens_mat - 2) & (
            (start_indexes_ex > middle_indexes_mat) | (end_indexes_ex < middle_indexes_mat))
    span_masks = torch.ones(X.shape[0], start_indexes.shape[0], device=args.device) * 1e6 * (~span_masks_bool)

    return X, y, start_indexes, end_indexes, attn_mask, span_masks, lens


class SNLIDataset(Dataset):
    def __init__(self, path, tokenizer):
        super().__init__()
        label_map = {"contradiction": 0, 'neutral': 1, "entailment": 2}
        self.data = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            if line_json['gold_label'] not in label_map:
                continue
            sentence1 = line_json['sentence1']
            sentence2 = line_json['sentence2']
            if sentence1.endswith("."):
                sentence1 = sentence1[:-1]
            if sentence2.endswith("."):
                sentence2 = sentence2[:-1]
            input_ids1 = tokenizer.encode(sentence1, add_special_tokens=False)
            input_ids2 = tokenizer.encode(sentence2, add_special_tokens=False)
            self.data.append((input_ids1 + [2] + input_ids2, label_map[line_json['gold_label']]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
