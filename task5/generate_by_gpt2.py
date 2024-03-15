# -*- coding: utf-8 -*-
# file: generate_by_gpt2.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import torch
import argparse
from transformers import BertTokenizer, GPT2LMHeadModel
from transformers.generation import GenerationConfig
from random import randint
import warnings


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


# 生成随机诗词
def generate(model, args, max_len=64):
    s = ''
    generation_config = GenerationConfig(max_length=max_len, do_sample=True,
                                         eos_token_id=args.tokenizer.sep_token_id,
                                         pad_token_id=args.tokenizer.pad_token_id)
    input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
    s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                              skip_special_tokens=True)
    s = s.replace(' ', '')
    return s


# beamsearch随机的句子下一个词始终无法满足需求时，手动随机下一个topk的词
def generate_next_random_topk_word(s, model, args, top_k=10):
    tokens_dict = args.tokenizer.encode_plus(s, return_tensors='pt')
    input_ids = tokens_dict['input_ids'].to(args.device)
    attn_mask = tokens_dict['attention_mask'].to(args.device)
    output = model(input_ids, attention_mask=attn_mask).logits
    output = output[0, -1, :]
    sort_idx = torch.sort(output).indices
    top_k = 10
    word = ''
    while not is_chinese(word):
        word = args.tokenizer.decode(sort_idx[randint(0, top_k - 1)])
    return word


# 生成特定格式的随机古诗，比如五言律诗，k=5个字，n=4句(逗号一句)
def generate_format(model, args, k=5, n=4):
    s = ''
    l = (k + 1) * n
    end, end_pre, repeat = 0, 0, 0
    while end != l:
        generation_config = GenerationConfig(max_length=end + 20, do_sample=True,
                                             eos_token_id=args.tokenizer.sep_token_id,
                                             pad_token_id=args.tokenizer.pad_token_id)
        input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
        s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                                  skip_special_tokens=True)
        s = s.replace(' ', '')
        for i in range(end, min(l, len(s))):
            sentence_idx = i // (k + 1)
            sentence_word_idx = i % (k + 1)
            if sentence_word_idx == k:
                if sentence_idx % 2 == 0:
                    if s[i] != '，':
                        s = s[:i] + '，'
                        break
                else:
                    if s[i] != '。':
                        s = s[:i] + '。'
                        break
            else:
                if not is_chinese(s[i]):
                    s = s[:i]
                    break
        end = min(l, len(s))
        if end == end_pre:
            repeat += 1
        else:
            end_pre = end
            repeat = 0
        if repeat == 10:
            s += generate_next_random_topk_word(s, model, args)
            end = min(l, len(s))
        s = s[:end]
    s = s.replace('。', '。\n')
    return s


# 生成特定格式的随机藏头诗
def generate_acrostic_format(model, args, head, k=5, n=4):
    assert len(head) == n
    s = ''
    l = (k + 1) * n
    end, end_pre, repeat = 0, 0, 0
    while end != l:
        generation_config = GenerationConfig(max_length=end + 20, do_sample=True,
                                             eos_token_id=args.tokenizer.sep_token_id,
                                             pad_token_id=args.tokenizer.pad_token_id)
        input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
        s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                                  skip_special_tokens=True)
        s = s.replace(' ', '')
        for i in range(end, min(l, len(s))):
            sentence_idx = i // (k + 1)
            sentence_word_idx = i % (k + 1)
            if sentence_word_idx == 0:
                s = s[:i] + head[sentence_idx]
                break
            elif sentence_word_idx == k:
                if sentence_idx % 2 == 0:
                    if s[i] != '，':
                        s = s[:i] + '，'
                        break
                else:
                    if s[i] != '。':
                        s = s[:i] + '。'
                        break
            else:
                if not is_chinese(s[i]):
                    s = s[:i]
                    break
        end = min(l, len(s))
        if end == end_pre:
            repeat += 1
        else:
            end_pre = end
            repeat = 0
        if repeat == 10:
            s += generate_next_random_topk_word(s, model, args)
            end = min(l, len(s))
        s = s[:end]
    s = s.replace('。', '。\n')
    return s


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='raw', type=str)
    parser.add_argument('--model_name', default='GPT2-Chinese-poem', type=str)
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='try 0.1 ~ 0.3 for small dataset, 0.5 for large ones')
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')

    args = parser.parse_args()

    model_classes = {
        'GPT2-Chinese': GPT2LMHeadModel,
        'GPT2-Chinese-poem': GPT2LMHeadModel,

    }

    best_model_paths = {
        'GPT2-Chinese': 'state_dict/GPT2-Chinese_raw_test_perplexity_1.6585',

    }

    gpt2_names = {
        'GPT2-Chinese': 'uer/gpt2-distil-chinese-cluecorpussmall',
        'GPT2-Chinese-poem': 'uer/gpt2-chinese-poem',

    }

    gpt2_paths = {
        'uer/gpt2-distil-chinese-cluecorpussmall': {
            'config': '../../../pretrained/GPT2Config/models--uer--gpt2-distil-chinese-cluecorpussmall/snapshots/c98ef629a1ece266e9d9183add4cbe5d4b99c7d5',
            'tokenizer': '../../../pretrained/BertTokenizer/models--uer--gpt2-distil-chinese-cluecorpussmall/snapshots/c98ef629a1ece266e9d9183add4cbe5d4b99c7d5',
            'model': '../../../pretrained/GPT2Model/models--uer--gpt2-distil-chinese-cluecorpussmall/snapshots/c98ef629a1ece266e9d9183add4cbe5d4b99c7d5',
        },
        'uer/gpt2-chinese-poem': {
            'config': '../../../pretrained/GPT2Config/models--uer--gpt2-chinese-poem/snapshots/6335c88ef6a3362dcdf2e988577b7bafeda6052b',
            'tokenizer': '../../../pretrained/BertTokenizer/models--uer--gpt2-chinese-poem/snapshots/6335c88ef6a3362dcdf2e988577b7bafeda6052b',
            'model': '../../../pretrained/GPT2Model/models--uer--gpt2-chinese-poem/snapshots/6335c88ef6a3362dcdf2e988577b7bafeda6052b',
        },

    }

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    args.gpt2_name = gpt2_names[args.model_name]
    args.gpt2_tokenizer_path = gpt2_paths[args.gpt2_name]['tokenizer']
    args.gpt2_model_path = gpt2_paths[args.gpt2_name]['model']
    args.tokenizer = BertTokenizer.from_pretrained(args.gpt2_tokenizer_path)

    model = model_classes[args.model_name].from_pretrained(args.gpt2_model_path)
    if 'poem' not in args.model_name:
        args.best_model_path = best_model_paths[args.model_name]
        model.load_state_dict(torch.load(args.best_model_path))
    model = model.to(args.device)
    model.eval()

    print(generate(model, args))
    print()
    print(generate(model, args))
    print()
    print(generate_format(model, args))
    print(generate_format(model, args, k=7))
    print(generate_format(model, args, n=8))
    print(generate_format(model, args, k=7, n=8))
    print(generate_acrostic_format(model, args, '春夏秋冬'))
    print(generate_acrostic_format(model, args, '春夏秋冬', k=7))
    print(generate_acrostic_format(model, args, '上下左右东南西北', n=8))
    print(generate_acrostic_format(model, args, '上下左右东南西北', k=7, n=8))
