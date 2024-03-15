# -*- coding: utf-8 -*-
# file: generate.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import argparse
from random import randint
from models import LSTM
from data_utils import build_tokenizer


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


# 生成随机诗词
def generate(model, tokenizer, k=100, max_len=128):
    # 第一个词对topk采样
    out = model(torch.tensor(tokenizer.transform('')).reshape(1, -1))[0]
    out = out[0]
    sort_idx = torch.sort(out, descending=True).indices
    s = ''
    while not is_chinese(s):
        s = tokenizer.idx2word[sort_idx[randint(0, k - 1)].item()]
    for i in range(max_len - 1):
        out = model(torch.tensor(tokenizer.transform(s)).reshape(1, -1))[0]
        out = out[i + 1]
        out = torch.argmax(out, dim=-1)
        word = tokenizer.idx2word[out.item()]
        if word == '[EOS]':
            break
        s += word
    return s


# 生成特定格式的随机古诗，比如五言律诗，k=5个字，n=4句(逗号一句)
def generate_format(model, tokenizer, k=5, n=4, top_k=100):
    s = ''
    for i in range(n):
        for j in range(k):
            out = model(torch.tensor(tokenizer.transform(s)).reshape(1, -1))[0]
            out = out[len(s)]
            sort_idx = torch.sort(out, descending=True).indices
            word = ''
            if i == 0 and j == 0:
                while not is_chinese(word):
                    word = tokenizer.idx2word[sort_idx[randint(0, top_k - 1)].item()]
            else:
                for idx in sort_idx:
                    word = tokenizer.idx2word[idx.item()]
                    if is_chinese(word):
                        break
            s += word
        if i % 2 == 0:
            s += '，'
        else:
            s += '。\n'
    return s


# 生成特定格式的随机藏头诗，topk=0生成固定藏头诗
def generate_acrostic_format(model, tokenizer, head, k=5, n=4, top_k=10):
    assert len(head) == n
    s = ''
    for i in range(n):
        for j in range(k):
            if j == 0:
                s += head[i]
                continue
            out = model(torch.tensor(tokenizer.transform(s)).reshape(1, -1))[0]
            out = out[len(s)]
            sort_idx = torch.sort(out, descending=True).indices
            word = ''
            if j == 1:
                while not is_chinese(word):
                    word = tokenizer.idx2word[sort_idx[randint(0, top_k - 1)].item()]
            else:
                for idx in sort_idx:
                    word = tokenizer.idx2word[idx.item()]
                    if is_chinese(word):
                        break
            s += word
        if i % 2 == 0:
            s += '，'
        else:
            s += '。\n'
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='raw', type=str)
    parser.add_argument('--model_name', default='LSTM', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='try 0.1 ~ 0.3 for small dataset, 0.5 for large ones')
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')

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

    print(generate(model, tokenizer))
    print()
    print(generate(model, tokenizer))
    print()
    print(generate_format(model, tokenizer))
    print(generate_format(model, tokenizer, k=7))
    print(generate_format(model, tokenizer, n=8))
    print(generate_format(model, tokenizer, k=7, n=8))
    print(generate_acrostic_format(model, tokenizer, '春夏秋冬'))
    print(generate_acrostic_format(model, tokenizer, '春夏秋冬', k=7))
    print(generate_acrostic_format(model, tokenizer, '上下左右东南西北', n=8))
    print(generate_acrostic_format(model, tokenizer, '上下左右东南西北', k=7, n=8))
