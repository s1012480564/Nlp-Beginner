# -*- coding: utf-8 -*-
# file: generate.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import argparse
import random
from models import LSTM
from data_utils import build_tokenizer


def generate(model, max_len=128):
    s = tokenizer.idx2word[random.randint(3, tokenizer.vocab_size - 1)]
    for i in range(max_len - 1):
        out = model(torch.tensor(tokenizer.transform(s)).reshape(1, -1))[0]
        out = torch.argmax(out, dim=-1)
        word = tokenizer.idx2word[out[i + 1].item()]
        if word == '[EOS]':
            break
        s += word
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='raw', type=str)
    parser.add_argument('--model_name', default='LSTM', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='try 0.1 ~ 0.3 for small dataset, 0.5 for large ones')
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.best_model_path = 'state_dict/LSTM_raw_test_perplexity_39.8594'
    args.model_classes = {
        'LSTM': LSTM,
    }
    tokenizer = build_tokenizer('', max_seq_len=args.max_seq_len,
                                dat_fname='pre_train/{0}_tokenizer.dat'.format(args.dataset))
    args.vocab_size = tokenizer.vocab_size

    model = args.model_classes[args.model_name](args)
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()

    print(generate(model))
    print(generate(model))
    print(generate(model))
    print(generate(model))
    print(generate(model))

