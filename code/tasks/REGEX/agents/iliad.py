import os
import sys
import json
import logging
import random
import re

import torch
import torch.nn as nn
import torch.distributions as D


from .executor import Executor

import models


class IliadStudent(Executor):

    def __init__(self, config):

        self.config = config
        self.device = config.device

        self.task_vocab = config.word_vocab
        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.regex_vocab

        model_config = self.config.student.model
        model_config.device = config.device
        model_config.task_vocab_size = len(self.task_vocab)
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.task_pad_idx = self.task_vocab['<PAD>']
        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)

        self.random = random
        self.random.seed(123)

        unlabeled_data_file = os.path.join(
            config.data_dir, 'unlabeled_regexes.txt')
        with open(unlabeled_data_file) as f:
            data = f.readlines()
            self.unlabeled_data = [list('<' + x.rstrip() + '>') for x in data]

        self.exploration_model = models.load(model_config).to(self.device)
        self.execution_model   = models.load(model_config).to(self.device)

        logging.info('exploration model: ' + str(self.exploration_model))
        logging.info('execution model: '   + str(self.execution_model))

        self.optim = torch.optim.Adam(
            list(self.exploration_model.parameters()) +
            list(self.execution_model.parameters()),
            lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

    def set_model(self, name):
        if name == 'exploration':
            self.model = self.exploration_model
        else:
            assert name == 'execution', name
            self.model = self.execution_model

    def reset(self):
        self.train_src_words = []
        self.train_gold_instructions = []
        self.train_instructions = []
        self.train_regexes = []

    def init(self, model_name, srcs, instructions, is_eval):

        self.set_model(model_name)
        super(IliadStudent, self).init(srcs, instructions, is_eval)

    def receive(self, src_words, gold_instructions, instructions, regexes):
        self.train_src_words.extend(src_words)
        self.train_gold_instructions.extend(gold_instructions)
        self.train_instructions.extend(instructions)
        self.train_regexes.extend(regexes)

    def act(self, gold_actions=None, sample=False, debug=False):

        action_logits = self.model.decode(self.prev_actions)
        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample()
                d = action_logits.softmax(dim=1)
            else:
                pred_actions = action_logits.max(dim=1)[1]
            self.prev_actions = pred_actions
            pred_actions = pred_actions.tolist()
            for i in range(self.batch_size):
                if not self.terminated[i]:
                    w = self.tgt_vocab.get(pred_actions[i])
                    self.pred_action_seqs[i].append(w)
        else:
            gold_actions = [self.tgt_vocab[w] for w in gold_actions]
            gold_actions = self._to_tensor(gold_actions).long()
            self.gold_action_seqs.append(gold_actions)
            self.prev_actions = gold_actions

        self.timer -= 1

        for i in range(self.batch_size):
            self.terminated[i] |= self.timer <= 0
            self.terminated[i] |= self.prev_actions[i].item() == self.tgt_vocab['>']

    def predict(self, src_words, instructions, model_name='execution', sample=False):

        self.init(model_name, src_words, instructions, True)
        while not self.has_terminated():
            self.act(sample=sample)

        return self.get_action_seqs()

    def compute_supervised_loss(self, model_name):

        batch_size = len(self.train_src_words)

        if batch_size == 0:
            return 0

        self.init(model_name, self.train_src_words, self.train_instructions, False)

        t = 0
        golds = [None] * batch_size
        while not self.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(self.train_regexes[i]):
                    golds[i] = self.train_regexes[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            self.act(gold_actions=golds)
            t += 1

        loss = super(IliadStudent, self).compute_loss()

        return loss

    def compute_unsupervised_loss(self, model_name):

        batch_size = len(self.train_src_words)

        if batch_size == 0:
            return 0

        regexes = self.random.sample(self.unlabeled_data, batch_size)

        instructions = self.train_instructions

        self.init(model_name, self.train_src_words, instructions, False)

        t = 0
        golds = [None] * batch_size
        while not self.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(regexes[i]):
                    golds[i] = regexes[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            self.act(gold_actions=golds)
            t += 1

        loss = super(IliadStudent, self).compute_loss()

        return loss

    def learn(self, unsup_weight):

        assert len(self.train_src_words) == len(self.train_instructions) == len(self.train_regexes)

        exp_loss = (1 - unsup_weight) * self.compute_supervised_loss('exploration') + \
                    unsup_weight * self.compute_unsupervised_loss('exploration')
        exe_loss = self.compute_supervised_loss('execution')

        loss = exp_loss + exe_loss

        if isinstance(loss, int) or isinstance(loss, float):
            return 0

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.gold_action_seqs)

    def save(self, name, trajectories=None):
        file_path = '%s/%s' % (self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'exp_model_state_dict': self.exploration_model.state_dict(),
                 'exe_model_state_dict': self.execution_model.state_dict(),
                 'optim_state_dict'    : self.optim.state_dict()
               }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.exploration_model.load_state_dict(ckpt['model_state_dict'])
            self.execution_model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.exploration_model.load_state_dict(ckpt['exp_model_state_dict'])
            self.execution_model.load_state_dict(ckpt['exe_model_state_dict'])
        logging.info('Loaded model from %s' % file_path)

        if hasattr(self.config.student, 'not_load_optim') and self.config.student.not_load_optim:
            return

        if 'unsupervised' not in file_path and 'last.ckpt' in file_path:
            self.optim.load_state_dict(ckpt['optim_state_dict'])
            logging.info('Loaded optim from %s' % file_path)

