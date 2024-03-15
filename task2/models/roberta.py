# -*- coding: utf-8 -*-
# file: roberta.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2024. All Rights Reserved.

import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


class ROBERTA(nn.Module):
    def __init__(self, args):
        super(ROBERTA, self).__init__()
        self.roberta_config = RobertaConfig.from_pretrained(args.roberta_config_path)
        self.roberta = RobertaModel.from_pretrained(args.roberta_model_path)
        self.fc = nn.Linear(self.roberta_config.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X, attn_mask):
        X = self.roberta(X, attn_mask).pooler_output  # [B,E]
        out = self.fc(self.dropout(X))
        return out
