# -*- coding: utf-8 -*-
# file: main4bert.py
# author: songanyang <s1012480564@foxmail.com>
# Copyright (C) 2024. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_utils4bert import data_collator, SNLIDataset
from models import ExplainableModel
from functools import partial

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, args):
        self.args = args
        self.model = args.model_class(args).to(args.device)

        self.trainset = SNLIDataset(args.dataset_file['train'], args.tokenizer)
        self.testset = SNLIDataset(args.dataset_file['test'], args.tokenizer)

        assert 0 <= args.valset_ratio < 1
        if args.valset_ratio > 0:
            valset_len = int(len(self.trainset) * args.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

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
            if type(child) != RobertaModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.args.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = batch[0].to(self.args.device)
                targets = batch[1].to(self.args.device)
                start_indexes = batch[2].to(self.args.device)
                end_indexes = batch[3].to(self.args.device)
                attn_mask = batch[4].to(self.args.device)
                span_masks = batch[5].to(self.args.device)
                lens = batch[6].to(self.args.device)
                outputs, alpha_ij = self.model(inputs, start_indexes, end_indexes, attn_mask, span_masks)

                ce_loss = criterion(outputs, targets)
                reg_loss = self.args.lambd * (alpha_ij ** 2).sum(dim=1).mean()
                loss = ce_loss + reg_loss
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.args.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.args.model_name, self.args.dataset,
                                                               round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            # if i_epoch - max_val_epoch >= self.opt.patience:
            #     print('>> early stop.')
            #     break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = t_batch[0].to(self.args.device)
                t_targets = t_batch[1].to(self.args.device)
                t_start_indexes = t_batch[2].to(self.args.device)
                t_end_indexes = t_batch[3].to(self.args.device)
                t_attn_mask = t_batch[4].to(self.args.device)
                t_span_masks = t_batch[5].to(self.args.device)
                t_lens = t_batch[6].to(self.args.device)
                t_outputs, _ = self.model(t_inputs, t_start_indexes, t_end_indexes, t_attn_mask, t_span_masks)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(_params, lr=self.args.lr, weight_decay=self.args.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.batch_size, shuffle=True,
                                       collate_fn=partial(data_collator, args=self.args))
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.args.batch_size, shuffle=False,
                                      collate_fn=partial(data_collator, args=self.args))
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.batch_size, shuffle=False,
                                     collate_fn=partial(data_collator, args=self.args))

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Self-Explain-Roberta', type=str)
    parser.add_argument('--dataset', default='small', type=str)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--initializer', default='kaiming_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='try 0.1 ~ 0.3 for small dataset, 0.5 for large ones')
    parser.add_argument('--l2reg', default=1e-5, type=float, help='try 1e-5 for BERT, 1e-2 for others')
    parser.add_argument('--lambd', default=1.0, type=float)
    parser.add_argument('--epochs', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=8, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valset_ratio', default=0.0, type=float,
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
        'Self-Explain-Roberta': ExplainableModel,

    }

    dataset_files = {
        'raw': {
            'train': 'dataset/snli_1.0_train.jsonl',
            'val': 'dataset/snli_1.0_dev.jsonl',
            'test': 'dataset/snli_1.0_test.jsonl'
        },
        'small': {
            'train': 'dataset/snli_1.0_train_small.jsonl',
            'val': 'dataset/snli_1.0_dev.jsonl',
            'test': 'dataset/snli_1.0_test.jsonl'
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

    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    args.roberta_config_path = '../../../pretrained/RobertaConfig/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    args.roberta_model_path = '../../../pretrained/RobertaModel/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    args.roberta_tokenizer_path = '../../../pretrained/RobertaTokenizer/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    args.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_tokenizer_path)

    # log_file = 'log/{}-{}-{}.log'.format(args.model_name, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_file = 'log/{}-{}.log'.format(args.model_name, args.dataset)
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()
