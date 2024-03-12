# -*- coding: utf-8 -*-
# file: cnn.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(args.hidden_dim * 2, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        lens = torch.sum(X != 0, dim=-1).to('cpu')
        X = self.embed(X)  # [B,L,E]
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(X_pack)  # [2,B,E]
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)  # [B,2E]
        out = self.fc(self.dropout(h_n))
        return out
