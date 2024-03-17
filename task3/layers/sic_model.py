# -*- coding: utf-8 -*-
# file: sic_model.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super(SICModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.fc1(hidden_states)  # B,L,E
        W2_h = self.fc2(hidden_states)
        W3_h = self.fc3(hidden_states)
        W4_h = self.fc4(hidden_states)

        W1_hi = torch.index_select(W1_h, dim=1, index=start_indexs)  # B,S,E  S:span_num
        W2_hj = torch.index_select(W2_h, dim=1, index=end_indexs)
        W3_hi = torch.index_select(W3_h, dim=1, index=start_indexs)
        W3_hj = torch.index_select(W3_h, dim=1, index=end_indexs)
        W4_hi = torch.index_select(W4_h, dim=1, index=start_indexs)
        W4_hj = torch.index_select(W4_h, dim=1, index=end_indexs)

        H_ij = torch.tanh(W1_hi + W2_hj + (W3_hi - W3_hj) + W4_hi * W4_hj)  # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        return H_ij
