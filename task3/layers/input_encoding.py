# -*- coding: utf-8 -*-
# file: input_encoding.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
from .bilstm import BiLSTM


class InputEncoding(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(InputEncoding, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(args.dropout)
        self.bilstm = BiLSTM(args)

    def forward(self, X, lens):
        lens = torch.sum(X != 0, dim=-1).to('cpu')
        X = self.embed(X)  # [B, L, E]
        X = self.dropout(X)
        out = self.bilstm(X, lens)  # [B, L, E]
        return out
