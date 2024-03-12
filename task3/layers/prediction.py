# -*- coding: utf-8 -*-
# file: prediction.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(self, args):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * args.max_seq_len, args.max_seq_len),
            nn.Tanh(),
            nn.Linear(args.max_seq_len, args.num_classes)
        )

    def forward(self, A, B):
        A_avg = F.avg_pool1d(A, A.shape[2]).squeeze(-1)  # [B,LA]
        A_max = F.max_pool1d(A, A.shape[2]).squeeze(-1)  # [B,LA]
        B_avg = F.avg_pool1d(B, B.shape[2]).squeeze(-1)  # [B,LB]
        B_max = F.max_pool1d(B, B.shape[2]).squeeze(-1)  # [B,LB]
        V = torch.cat((A_avg, A_max, B_avg, B_max), dim=-1)
        out = self.mlp(V)
        return out
