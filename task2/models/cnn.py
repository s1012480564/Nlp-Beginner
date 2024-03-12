# -*- coding: utf-8 -*-
# file: cnn.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):  # https://arxiv.org/pdf/1408.5882.pdf
    def __init__(self, embedding_matrix, args):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=args.kernel_num,
                      kernel_size=(k, args.embed_dim))
            for k in args.kernel_sizes
        ])
        self.fc = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        X = self.embed(X)  # [B, L, E]
        X = X.unsqueeze(1)  # [B, C=1, L, E]
        Xs = [F.relu(conv(X)).squeeze(3) for conv in self.convs]  # [B, C, L - k + 1] for each
        Xs = [F.max_pool1d(X, X.shape[2]).squeeze(2) for X in Xs]  # [B, C] for each
        X = self.dropout(torch.cat(Xs, dim=1))  # [B,C*len(ks)]
        out = self.fc(X)
        return out
