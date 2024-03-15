# -*- coding: utf-8 -*-
# file: bert.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2024. All Rights Reserved.

import torch.nn as nn
from transformers import BertConfig, BertModel


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_config_path)
        self.bert = BertModel.from_pretrained(args.bert_model_path)
        self.fc = nn.Linear(self.bert_config.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X, attn_mask):
        X = self.bert(X, attn_mask).pooler_output  # [B,E]
        out = self.fc(self.dropout(X))
        return out
