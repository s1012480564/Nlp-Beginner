# -*- coding: utf-8 -*-
# file: bert_crf.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

from torch import nn
from transformers import BertConfig, BertModel
from TorchCRF import CRF


class BERT_CRF(nn.Module):
    def __init__(self, args):
        super(BERT_CRF, self).__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_config_path)
        self.bert = BertModel.from_pretrained(args.bert_model_path)
        self.fc = nn.Linear(self.bert_config.hidden_size, args.target_size)
        self.crf = CRF(args.target_size)

    def encode_layer(self, X, attn_mask):
        X = self.bert(X, attn_mask).last_hidden_state  # [B,L,E]
        out = self.fc(X)
        return out

    def forward(self, X, targets, attn_mask, mask):
        X = self.encode_layer(X, attn_mask)[:, 1:, :]  # remove [CLS]
        out = self.crf(X, targets, mask)
        return out

    def predict(self, X, attn_mask, mask):
        X = self.encode_layer(X, attn_mask)[:, 1:, :]
        out = self.crf.viterbi_decode(X, mask)
        return out
