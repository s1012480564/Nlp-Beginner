# -*- coding: utf-8 -*-
# file: esim.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from task3.layers import InputEncoding, LocalInferenceModeling, InferenceComposition, Prediction


class ESIM(nn.Module):  # https://arxiv.org/pdf/1609.06038v3.pdf
    def __init__(self, embedding_matrix, args):
        super(ESIM, self).__init__()
        self.input_encoding = InputEncoding(embedding_matrix, args)
        self.local_inference = LocalInferenceModeling(embedding_matrix, args)
        self.inference_composition = InferenceComposition(args)
        self.prediction = Prediction(args)

    def forward(self, A, B):
        lens_A = torch.sum(A != 0, dim=-1).to('cpu')
        lens_B = torch.sum(B != 0, dim=-1).to('cpu')
        A = self.input_encoding(A, lens_A)
        B = self.input_encoding(B, lens_B)
        m_A, m_B = self.local_inference(A, B)
        v_A = self.inference_composition(m_A, lens_A)
        v_B = self.inference_composition(m_B, lens_B)
        out = self.prediction(v_A, v_B)
        return out
