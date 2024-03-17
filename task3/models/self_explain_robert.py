# -*- coding: utf-8 -*-
# file: self_explain_robert.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from task3.layers import SICModel, InterpretationModel


class ExplainableModel(nn.Module):  # https://arxiv.org/pdf/2012.01786.pdf
    def __init__(self, args):
        super(ExplainableModel, self).__init__()
        self.roberta_config = RobertaConfig.from_pretrained(args.roberta_config_path)
        self.intermediate = RobertaModel.from_pretrained(args.roberta_model_path)
        self.span_info_collect = SICModel(self.roberta_config.hidden_size)
        self.interpretation = InterpretationModel(self.roberta_config.hidden_size)
        self.fc = nn.Linear(self.roberta_config.hidden_size, args.num_classes)

    def forward(self, input_ids, start_indexes, end_indexes, attn_mask, span_masks):
        H = self.intermediate(input_ids, attention_mask=attn_mask).last_hidden_state  # B,L,E
        H_ij = self.span_info_collect(H, start_indexes, end_indexes)  # B,S,E  S:span_num
        H, alpha_ij = self.interpretation(H_ij, span_masks)  # B,E
        out = self.fc(H)
        return out, alpha_ij
