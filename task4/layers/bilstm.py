# -*- coding: utf-8 -*-
# file: lstm.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.max_seq_len = args.max_seq_len
        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_dim // 2, args.lstm_num_layers, batch_first=True,
                              bidirectional=True)

    def forward(self, X, lens):
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)  # [B, L, E]
        out, (_, _) = self.bilstm(X_pack)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.max_seq_len)  # [B, L, E]
        return out
