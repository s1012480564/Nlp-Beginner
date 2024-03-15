# -*- coding: utf-8 -*-
# file: main4gpt2.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy as np
from time import strftime, localtime
from transformers import GPT2Config, BertTokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader, random_split
from data_utils4gpt2 import data_collator, PoetryDataset
from functools import partial

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, args):
        self.args = args
        self.model = args.model_class.from_pretrained(args.gpt2_model_path).to(args.device)
        self.trainset = PoetryDataset(args.dataset_file['train'], args.tokenizer)

        assert 0 <= args.valset_ratio < 1
        if args.valset_ratio > 0:
            valset_len = int(len(self.trainset) * args.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        # else:
        #     self.valset = self.testset

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.args):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != GPT2LMHeadModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, optimizer, train_data_loader, val_data_loader):
        min_val_perplexity = 2e9
        min_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.args.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_total, loss_total, perplexity = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = batch[0].to(self.args.device)
                attn_mask = batch[1].to(self.args.device)
                lens = batch[2]
                loss = self.model(inputs, labels=inputs, attention_mask=attn_mask).loss

                loss.backward()
                optimizer.step()

                num = torch.sum(lens).item()
                n_total += num
                loss_total += loss.item() * num
                if global_step % self.args.log_step == 0:
                    train_loss = loss_total / n_total
                    perplexity = math.exp(train_loss)
                    logger.info('loss: {:.4f}, perplexity: {:.4f}'.format(train_loss, perplexity))
            if self.args.valset_ratio == 0:
                if (i_epoch + 1) % 5 == 0:
                    train_loss = loss_total / n_total
                    perplexity = math.exp(train_loss)
                    if perplexity < min_val_perplexity:
                        min_val_perplexity = perplexity
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = 'state_dict/{0}_{1}_test_perplexity_{2}'.format(self.args.model_name, self.args.dataset,
                                                                               round(perplexity, 4))
                        torch.save(self.model.state_dict(), path)
                        logger.info('>> saved: {}'.format(path))
                continue
            val_perplexity = self._evaluate_perplexity(val_data_loader)
            logger.info('> val_perplexity: {:.4f}'.format(val_perplexity))
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_perplexity_{2}'.format(self.args.model_name, self.args.dataset,
                                                                      round(val_perplexity, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            # if i_epoch - min_val_epoch >= self.opt.patience:
            #     print('>> early stop.')
            #     break

        return path

    def _evaluate_perplexity(self, data_loader):
        t_loss_total, n_total = 0, 0
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = t_batch[0].to(self.args.device)
                t_attn_mask = t_batch[1].to(self.args.device)
                t_lens = t_batch[2]

                t_loss = self.model(t_inputs, labels=t_inputs, attention_mask=t_attn_mask).loss

                num = torch.sum(t_lens).item()
                t_loss_total += t_loss.item() * num
                n_total += num

        val_loss = t_loss_total / n_total
        perplexity = math.exp(val_loss)
        return perplexity

    def run(self):
        # Loss and Optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(_params, lr=self.args.lr, weight_decay=self.args.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.batch_size, shuffle=True,
                                       collate_fn=partial(data_collator, args=self.args))
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.args.batch_size, shuffle=False,
        #                                        collate_fn=partial(data_collator, args=self.args))
        val_data_loader = None
        if self.args.valset_ratio:
            val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.batch_size, shuffle=False,
                                         collate_fn=partial(data_collator, args=self.args))

        self._reset_params()
        best_model_path = self._train(optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        # test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        # logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='GPT2-Chinese-poem', type=str)
    parser.add_argument('--dataset', default='raw', type=str)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--initializer', default='kaiming_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for GPT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='try 0.1 ~ 0.3 for small dataset, 0.5 for large ones')
    parser.add_argument('--l2reg', default=0.01, type=float, help='try 1e-5 for GPT, 1e-2 for others')
    # gpt-chinese epochs=13的时候验证困惑度最好
    # 生成实验发现，epochs较少时，写的不太像诗词。而epochs在10个左右时，生成的句子经常不正常，甚至空白
    # epochs再多一些，写出来的比较像词，但是单句很长，还是不太正常
    # 过拟合时，生成的诗词很好，并且并没有像LSTM那样几乎完全抄的数据集的句子。但是严重过拟合生成的诗词还是会和数据集相似度很高
    # 最终采用的是 epochs=40 的参数
    # gpt-chinese-poem epochs=11的时候验证困惑度最好
    # 毕竟现成的模型，只训练1个epoch都发现效果明显变差了，所以直接采用原始的参数......
    parser.add_argument('--epochs', default=20, type=int, help='try larger number for non-GPT models')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='try 16, 32, 64 for GPT models')  # total_size = 164
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    model_classes = {
        'GPT2-Chinese': GPT2LMHeadModel,
        'GPT2-Chinese-poem': GPT2LMHeadModel,

    }

    dataset_files = {
        'raw': {
            'train': 'dataset/poetryFromTang.txt',
        },
    }

    initializers = {
        'kaiming_uniform_': torch.nn.init.kaiming_uniform_,
        'kaiming_normal_': torch.nn.init.kaiming_normal_,
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'Adadelta': torch.optim.Adadelta,  # default lr=1.0
        'Adagrad': torch.optim.Adagrad,  # default lr=0.01
        'Adam': torch.optim.Adam,  # default lr=0.001
        'Adamax': torch.optim.Adamax,  # default lr=0.002
        'AdamW': torch.optim.AdamW,  # default lr=0.001
        'ASGD': torch.optim.ASGD,  # default lr=0.01
        'RMSprop': torch.optim.RMSprop,  # default lr=0.01
        'SGD': torch.optim.SGD,
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

    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    args.gpt2_name = gpt2_names[args.model_name]
    args.gpt2_config_path = gpt2_paths[args.gpt2_name]['config']
    args.gpt2_tokenizer_path = gpt2_paths[args.gpt2_name]['tokenizer']
    args.gpt2_model_path = gpt2_paths[args.gpt2_name]['model']
    args.tokenizer = BertTokenizer.from_pretrained(args.gpt2_tokenizer_path)

    # log_file = 'log/{}-{}-{}.log'.format(args.model_name, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_file = 'log/{}-{}'.format(args.model_name, args.dataset)
    if args.valset_ratio:
        log_file += '_val'
    else:
        log_file += '_train'
    log_file += '.log'

    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()
