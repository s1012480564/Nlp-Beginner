# -*- coding: utf-8 -*-
# file: lstm_crf.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
from TorchCRF import CRF
from task4.layers import BiLSTM


class LSTM_CRF(nn.Module):  # https://arxiv.org/pdf/1603.01360.pdf
    def __init__(self, embedding_matrix, args):
        super(LSTM_CRF, self).__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bilstm = BiLSTM(args)
        self.fc = nn.Linear(args.hidden_dim, args.target_size)
        self.crf = CRF(args.target_size)

    def encode_layer(self, X):
        lens = torch.sum(X != 0, dim=-1).to('cpu')
        X = self.embed(X)
        X = self.bilstm(X, lens)
        out = self.fc(X)
        return out

    def forward(self, X, targets):
        mask = (X != 0)
        X = self.encode_layer(X)
        out = self.crf(X, targets, mask)
        return out

    def predict(self, X):
        mask = (X != 0)
        X = self.encode_layer(X)
        out = self.crf.viterbi_decode(X, mask)
        return out
