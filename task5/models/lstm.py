# -*- coding: utf-8 -*-
# file: cnn.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = nn.Embedding(args.vocab_size, args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, batch_first=True)
        self.fc = nn.Linear(args.hidden_dim, args.vocab_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        lens = torch.sum(X != 0, dim=-1).to('cpu')
        X = self.embed(X)
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        X_pack, (_, _) = self.lstm(X_pack)
        X, _ = pad_packed_sequence(X_pack, batch_first=True, total_length=self.max_seq_len)
        out = self.fc(self.dropout(X))
        return out
