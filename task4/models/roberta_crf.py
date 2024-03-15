# -*- coding: utf-8 -*-
# file: roberta_crf.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

from torch import nn
from transformers import RobertaConfig, RobertaModel
from TorchCRF import CRF


class ROBERTA_CRF(nn.Module):
    def __init__(self, args):
        super(ROBERTA_CRF, self).__init__()
        self.roberta_config = RobertaConfig.from_pretrained(args.roberta_config_path)
        self.roberta = RobertaModel.from_pretrained(args.roberta_model_path)
        self.fc = nn.Linear(self.roberta_config.hidden_size, args.target_size)
        self.crf = CRF(args.target_size)

    def encode_layer(self, X, attn_mask):
        X = self.roberta(X, attn_mask).last_hidden_state  # [B,L,E]
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
