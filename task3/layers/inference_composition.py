# -*- coding: utf-8 -*-
# file: inference_composition.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

from torch import nn
import torch.nn.functional as F
from .bilstm import BiLSTM


class InferenceComposition(nn.Module):
    def __init__(self, args):
        super(InferenceComposition, self).__init__()
        self.fc = nn.Linear(4 * args.hidden_dim, args.hidden_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.bilstm = BiLSTM(args)

    def forward(self, X, lens):
        X = F.relu(self.fc(X))
        X = self.dropout(X)
        out = self.bilstm(X, lens)
        return out
