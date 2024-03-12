# -*- coding: utf-8 -*-
# file: local_inference_modeling.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F


class LocalInferenceModeling(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(LocalInferenceModeling, self).__init__()

    def forward(self, A, B):
        score = A @ (B.transpose(1, 2))  # [B,LA,LB]
        scoreA = F.softmax(score, dim=2)
        scoreB = F.softmax(score, dim=1)
        A_t = scoreA @ B  # [B,LA,E]
        B_t = scoreB.transpose(1, 2) @ A  # [B,LB,E]
        out_A = torch.cat([A, A_t, A - A_t, A * A_t], dim=-1)  # [B,LA,4E]
        out_B = torch.cat([B, B_t, B - B_t, B * B_t], dim=-1)  # [B,LB,4E]
        return out_A, out_B
